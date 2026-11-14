# Learning Topology-Aware Dynamic Associations for Robust Multi-Person Pose Estimation

<p align="center">
  <strong>English</strong> | <a href="README_zh-CN.md">简体中文</a>
</p>

## Method

We present **TopoDA**, an end-to-end framework that learns topology-aware dynamic associations for multi-person pose estimation. It combines locally-aware feature alignment (LAFA), dynamic keypoint box modulation (DKBM), and topology-aware refinement in a coarse-to-fine pipeline. By aligning multi-scale features, adapting keypoint regions, and modeling joint relations within and across people, TopoDA addresses pose ambiguity in crowded and occluded scenes.

![TopoDA](figs/overview.jpg)

## Environment

### Installation

This release uses `Python 3.11`, `PyTorch 2.7.1`, `torchvision 0.22.1`, and `CUDA 12.8`. Build the CUDA operators on Linux with CUDA Toolkit 12.8 (`nvcc`) and a C++ compiler.

1. Create and activate a Conda environment.

   ```bash
   conda create -n TopoDA python=3.11 -y
   conda activate TopoDA
   ```

2. Install PyTorch and torchvision.

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
   ```

3. Clone this repository.

   ```bash
   git clone https://github.com/suiyingliuxin/TopoDA.git
   cd TopoDA
   ```

4. Install the remaining dependencies and the COCO API.

   ```bash
   python -m pip install -r requirements.txt
   ```

   `requirements.txt` includes `pycocotools`. The CrowdPose API is included in `crowdposetools/`.

5. Compile DCNv4 and multi-scale deformable attention.

   ```bash
   bash scripts/build_ops.sh
   ```

## Data preparation

### CrowdPose

Download [CrowdPose](https://github.com/jeffffffli/CrowdPose#dataset) and organize it as follows:

```text
data/crowdpose/
├── images/
└── json/
    ├── crowdpose_train.json
    ├── crowdpose_val.json
    ├── crowdpose_trainval.json
    └── crowdpose_test.json
```

Alternatively, use the download script:

```bash
python -m pip install -r requirements-data.txt
python scripts/download_crowdpose.py --root data/crowdpose
```

If `crowdpose_trainval.json` is missing, run `python scripts/merge_crowdpose.py data/crowdpose`.

### COCO

Download the images and person-keypoint annotations from [COCO 2017](https://cocodataset.org/#download):

```text
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── person_keypoints_train2017.json
    └── person_keypoints_val2017.json
```

## Run

The multi-GPU examples use four GPUs on one machine; set `--nproc_per_node` to your GPU count. `batch_size=2` means two images per GPU, for a total batch size of eight on four GPUs. EMA is enabled in the multi-GPU training and resume examples.

### Backbone weights

Download the Swin-T ImageNet initialization:

```bash
mkdir -p weights
curl -L --fail --retry 3 \
  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth \
  -o weights/swin_tiny_patch4_window7_224.pth
```

### Training on CrowdPose

**Single GPU**

```bash
python main.py -c config/topoda_crowdpose_swint.py \
  --crowdpose_path data/crowdpose \
  --backbone_weights weights/swin_tiny_patch4_window7_224.pth \
  --output_dir runs/crowdpose_swint --num_workers 6 --amp --eval_split test \
  --options crowdpose_train_split=trainval
```

**Four GPUs on one machine**

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
  -c config/topoda_crowdpose_swint.py \
  --crowdpose_path data/crowdpose \
  --backbone_weights weights/swin_tiny_patch4_window7_224.pth \
  --output_dir runs/crowdpose_swint_4gpu --num_workers 4 \
  --amp --eval_split test --find_unused_params \
  --options crowdpose_train_split=trainval batch_size=2 use_ema=True
```

For ResNet-50, use `config/topoda_crowdpose_r50.py` and omit `--backbone_weights`.

### Resume training

**Single GPU**

```bash
python main.py -c config/topoda_crowdpose_swint.py \
  --crowdpose_path data/crowdpose --output_dir runs/crowdpose_swint \
  --resume runs/crowdpose_swint/checkpoint.pth --amp --eval_split test \
  --options crowdpose_train_split=trainval
```

**Four GPUs on one machine**

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
  -c config/topoda_crowdpose_swint.py \
  --crowdpose_path data/crowdpose --output_dir runs/crowdpose_swint_4gpu \
  --resume runs/crowdpose_swint_4gpu/checkpoint.pth \
  --amp --eval_split test --find_unused_params \
  --options crowdpose_train_split=trainval batch_size=2 use_ema=True
```

Keep the original configuration and split options when resuming.

### Evaluation on CrowdPose

**Single GPU**

```bash
python main.py -c config/topoda_crowdpose_swint.py \
  --crowdpose_path data/crowdpose --eval --eval_split test \
  --resume runs/crowdpose_swint/checkpoint_best.pth \
  --output_dir runs/crowdpose_eval --save_results --amp
```

**Four GPUs on one machine**

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
  -c config/topoda_crowdpose_swint.py \
  --crowdpose_path data/crowdpose --eval --eval_split test \
  --resume runs/crowdpose_swint_4gpu/checkpoint_best.pth \
  --output_dir runs/crowdpose_eval_4gpu --save_results --amp
```

For evaluation on val, use `--eval_split val` with the corresponding train/val checkpoint.

### Training and evaluation on COCO

**Single GPU**

```bash
# Train
python main.py -c config/TopoDA.cfg.py \
  --dataset_file coco --coco_path data/coco \
  --output_dir runs/coco_r50 --amp

# Evaluate
python main.py -c config/TopoDA.cfg.py \
  --dataset_file coco --coco_path data/coco --eval --eval_split val \
  --resume runs/coco_r50/checkpoint_best.pth \
  --output_dir runs/coco_eval --save_results --amp
```

**Four GPUs on one machine**

```bash
# Train
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
  -c config/TopoDA.cfg.py \
  --dataset_file coco --coco_path data/coco \
  --output_dir runs/coco_r50_4gpu --amp --find_unused_params \
  --options batch_size=2 use_ema=True

# Evaluate
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
  -c config/TopoDA.cfg.py \
  --dataset_file coco --coco_path data/coco --eval --eval_split val \
  --resume runs/coco_r50_4gpu/checkpoint_best.pth \
  --output_dir runs/coco_eval_4gpu --save_results --amp
```

### Image inference

```bash
python infer.py -c config/topoda_crowdpose_swint.py \
  --checkpoint runs/crowdpose_swint/checkpoint_best.pth \
  --input path/to/image_or_directory --output_dir runs/inference --amp
```

## Citation

If you find TopoDA useful, please consider giving this repository a ⭐ star and following its updates. If you use it in your research, please cite our paper:

```bibtex
@inproceedings{hu2026learning,
  title={Learning topology-aware dynamic associations for robust multi-person pose estimation},
  author={Hu, Shengnan and Liu, Yandong and Liu, Jiangnan and Chen, Yahong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={6},
  pages={4914--4922},
  year={2026}
}
```

