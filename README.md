# TASSO

This is the official implementation of paper [TASSO: TAsk-Specific Subspace Optimization for Continual Learning of Vision-Language Models](https://arxiv.org/abs/2608.21487). Our approach TASSO aims to mitigate both backward and forward forgetting while also restore plasticity for the CLIP model.

![PDF Preview](images/architecture.svg)

## Dataset Preparation

Download the datasets used in this project by following the instructions in the [CoOp dataset guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

Supported datasets:

- FGVCAircraft
- DTD
- EuroSAT
- Flowers102
- Food101
- OxfordPets
- StanfordCars
- UCF101
- ImageNet

Organize each dataset with the following structure:

```text
<DATASET_NAME>/
    images/
    <image folders or files>
    <DATASET_NAME>_annotations.json
```

The `<DATASET_NAME>_annotations.json` file should contain the train, validation, and test splits together with class names. The annotation files used in this project are available [here](https://drive.google.com/drive/folders/144OIxusHyB8tRtlnvVGttCx0UE0ab_bv?usp=sharing).

Before running any command, update `data.root` in the files under `configs/` so it points to your dataset root directory.

## Running the Code

### Train and Evaluate on One Dataset

Train on a single dataset and evaluate on all datasets with 4 GPUs:

```sh
python -m scripts.train_and_eval --config_path configs/snd_config_4_gpus.yaml --dataset fgvc-aircraft --distributed --nproc_per_node 4
```

Train on a single dataset with 1 GPU:

```sh
python -m scripts.train_and_eval --config_path configs/snd_config_1_gpu.yaml --dataset fgvc-aircraft --distributed --nproc_per_node 1
```

Continue training from a previously trained dataset:

```sh
python -m scripts.train_and_eval --config_path configs/snd_config_4_gpus.yaml --pretrained_dataset fgvc-aircraft --dataset dtd --distributed --nproc_per_node 4
```

### Continual Training on the Full Sequence

Run continual training and evaluation over the predefined dataset sequence:

```sh
python -m scripts.continually_train --config_path configs/snd_config_4_gpus.yaml --order 0 --distributed --nproc_per_node 4
```

### Inference

Run inference with a saved checkpoint:

```sh
python -m scripts.inference --model_path outputs/order_0/checkpoint_latest.pth
```
## Checkpoints
To reproduce the results in the paper, check the following link: [Checkpoints](https://drive.google.com/drive/folders/1kGEXxXaME9hfV8EORMT6LClzp2dgZNCg?usp=sharing)

## Citation
```
@article{sun2026tasso,
  title={TASSO: TAsk-Specific Subspace Optimization for Continual Learning of Vision-Language Models},
  author={Sun, Chang and Barbato, Francesco and Caligiuri, Matteo and Zanuttigh, Pietro},
  journal={arXiv preprint arXiv:2608.21487},
  year={2026}
}
```

## Reference

This project builds on the code contribution associated with the Select and Distill paper. For the original work, see [Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models](https://arxiv.org/abs/2403.09296).

