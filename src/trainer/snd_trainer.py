import torch
import torch.nn as nn
import torch.nn.functional as F
import math
##### for visualization
from pathlib import Path
import json
import torch.distributed as dist
from src.utils import is_main_process
from src.utils import main_process
from collections import deque
from src.datasets.utils import load_single_class_name_list

from scripts.utils import DEFAULT_DATASET_SEQ


from src.trainer.base_trainer import BaseKDTrainer
from src.trainer.utils import L2Loss
from src.trainer.utils import CosineLoss
from src.trainer.utils import GeodesicLoss
from src.trainer.utils import CosineLRScheduler, get_optimizer, orthonormalize_qr



class SnDTrainer(BaseKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_teacher_model.eval()
        self.feature_criterion = L2Loss(reduce=None, square=False)
        self.feature_criterion2 = CosineLoss(reduce=None, square=False)
        self.feature_criterion3 = GeodesicLoss(reduce=None, square=False)

        
        self.use_subspace_mkd = bool(self.method_config.params.get("use_subspace_mkd", True))
        if self.use_subspace_mkd:
            self._init_subspace_projector_choice_a()

    def _save_final_projector_A(self):
        if not getattr(self, "use_subspace_mkd", False):
            return

        unwrapped = self.unwrapped_model(self.train_model)
        if not hasattr(unwrapped.model, "subspace_projector_raw"):
            return

        A = unwrapped.model.subspace_projector_raw.detach().cpu()

        save_obj = {
            "A": A,  # raw learned matrix A
            "dataset": self.config.data.name,
            "task_idx": getattr(self, "task_idx", None),
            "order_idx": getattr(self, "order_idx", None),
            "shape": list(A.shape),
        }

        save_path = self.output_dir / "matrix_A.pth"
        print(f"Saving final projector A to {save_path}.")
        torch.save(save_obj, save_path)

    def _get_current_group_lrs(self):
        lr_dict = {}
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return lr_dict

        for i, g in enumerate(self.optimizer.param_groups):
            group_name = g.get("group_name", f"group_{i}")
            lr_dict[f"lr_{group_name}"] = float(g["lr"])
        return lr_dict

    @main_process
    def save(self, epoch=None):
        # Same as BaseTrainer.save, but drop the projector key so evaluation can load strictly.
        unwrapped_eval_model = self.unwrapped_model(self.eval_model)
        state_dict = unwrapped_eval_model.get_state_dict()

        # Remove the dynamically attached parameter
        state_dict.pop("subspace_projector_raw", None)

        # Extra safety
        for k in list(state_dict.keys()):
            if k.endswith("subspace_projector_raw"):
                state_dict.pop(k, None)

        save_obj = {"model": state_dict}

        is_final_task_save = epoch is None

        if not epoch:
            epoch = "latest"

        save_path = self.output_dir / f"checkpoint_{epoch}.pth"
        print(f"Saving checkpoint at epoch {epoch} to {save_path}.")
        torch.save(save_obj, save_path)

        # Save A only once, after finishing the current task training
        if is_final_task_save:
            self._save_final_projector_A()



    @property
    def prev_teacher_model(self):
        return self._teachers["prev"]    

    ##########################################################################################################
    def _init_subspace_projector_choice_a(self):
        projector_dim = int(self.method_config.params.get("projector_dim", 64))

        # Infer embedding dim d from text encoding
        unwrapped = self.unwrapped_model(self.train_model)
        with torch.no_grad():
            dummy_txt = unwrapped.encode(text=unwrapped.class_tokens[:1], normalize=True)
            d = int(dummy_txt.shape[-1])

        # initilization, random
        raw = torch.randn(d, projector_dim, device=dummy_txt.device, dtype=dummy_txt.dtype) * 0.02
        unwrapped.model.subspace_projector_raw = nn.Parameter(raw, requires_grad=True)

        # Rebuild optimizer to include the newly attached parameter.
        if self.training_mode:
            self.optimizer = get_optimizer(unwrapped, self.config.task)
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer, self.config.task, self.num_total_train_steps
            )

            if is_main_process():
                print("Optimizer param groups after adding projector:")
                for i, g in enumerate(self.optimizer.param_groups):
                    print(
                        f"  group {i} | name={g.get('group_name', 'NA')} | "
                        f"lr={g['lr']:.6e} | wd={g.get('weight_decay', 'NA')}"
                    )

    def _get_projector_basis(self) -> torch.Tensor:
        unwrapped = self.unwrapped_model(self.train_model)
        raw = unwrapped.model.subspace_projector_raw
        return orthonormalize_qr(raw)


    def _clip_logits(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        img = F.normalize(img_emb, dim=-1)
        txt = F.normalize(txt_emb, dim=-1)
        return (logit_scale * (img @ txt.t()))

    def _subspace_logits(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, U: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        z = img_emb @ U   # [B, k]
        q = txt_emb @ U   # [C, k]
        z = F.normalize(z, dim=-1)
        q = F.normalize(q, dim=-1)
        return (logit_scale * (z @ q.t()))

    def _project_complement(self, emb: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        z = emb @ U
        e_par = z @ U.t()
        return emb - e_par

    def _geodesic_from_embeddings(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        xh = F.normalize(x, dim=-1, eps=eps)
        yh = F.normalize(y, dim=-1, eps=eps)
        cos = (xh * yh).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        return torch.acos(cos)


    def snd_loss(
        self,
        images,
        labels,
        ratio_prev=3.0,
        label_smoothing=0.0,
        # ---- Subspace MKD (Method 1) params ----
        use_subspace_mkd=True,
        projector_dim=144,  # read in __init__ for Choice A
        alpha_subce=0.5,
        sub_logit_scale=10,
        geo_eps=1e-6,
        **_,
    ):
        if use_subspace_mkd is None:
            use_subspace_mkd = getattr(self, "use_subspace_mkd", False)

        if use_subspace_mkd:
            # Current-task supervised losses on (images, labels)
            img_emb, txt_emb, logit_scale = self.train_model(images, get_features=True)

            logits_full = self._clip_logits(img_emb, txt_emb, logit_scale)
            full_ce = F.cross_entropy(
                logits_full, labels, label_smoothing=label_smoothing
            )

            # Orthonormal basis U from raw projector (learned)
            U = self._get_projector_basis()

            # Subspace CLIP-style CE (task-relevant projector)
            if sub_logit_scale is not None:
                sub_scale = torch.as_tensor(
                    float(sub_logit_scale), device=logit_scale.device, dtype=logit_scale.dtype
                )
            else:
                sub_scale = logit_scale

            logits_sub = self._subspace_logits(img_emb, txt_emb, U, sub_scale)
            sub_ce = F.cross_entropy(
                logits_sub, labels, label_smoothing=label_smoothing
            )

            # KD losses on reference images
            ref_images, _ = self.get_ref_data(self.ref_loader)
            student_ref_img_emb, _, _ = self.train_model(ref_images, get_features=True)
            with torch.no_grad():
                prev_ref_img_emb, _, _ = self.prev_teacher_model(ref_images, get_features=True)

            # (1) L2 KD on projected coordinates (task-relevant subspace)
            z_s = student_ref_img_emb @ U
            z_t = prev_ref_img_emb @ U
            # kd_l2 = torch.pow(torch.norm(z_s - z_t, dim=-1), 2).mean()
            kd_relevant = self._geodesic_from_embeddings(z_s, z_t, eps=geo_eps).mean()


            # (2) Geodesic KD on complementary component
            e_s_perp = self._project_complement(student_ref_img_emb, U)
            e_t_perp = self._project_complement(prev_ref_img_emb, U)
            kd_irrelevant = self._geodesic_from_embeddings(e_s_perp, e_t_perp, eps=geo_eps).mean()
            # kd_irrelevant = torch.pow(torch.norm(e_s_perp - e_t_perp, dim=-1), 2).mean()

            # separate distillation on two subspaces
            kd_prev = kd_relevant + kd_irrelevant
            # kd_prev = self._get_kd_loss(
            #      student_ref_img_emb,
            #      prev_ref_img_emb,
            #      feature_criterion=self.feature_criterion3,
            #  ).mean()

            total = full_ce + (alpha_subce * sub_ce) + (ratio_prev * kd_prev)

            return total, {
                "loss": full_ce.item(),
                "sub_ce": sub_ce.item(),
                "prev_kd_relevant": kd_relevant.item(),
                "prev_kd_irrelevant": kd_irrelevant.item(),
                **self._get_current_group_lrs(),
            }