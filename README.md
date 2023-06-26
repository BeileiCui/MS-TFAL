# [MICCAI'2023] Multi-Scale Temporal Feature Affinity Learning for Robust Video Segmentation (MS-TFAL)

![Image](https://github.com/BeileiCui/MS-TFAL/blob/main/main_figure.jpg)
## Introduction
* This is the implementation for our MS-TFAL.

## Usage

### Requirements

* pytorch
* numpy
* opencv
* sklearn
* skimage
* tqdm
* tensorboardX

### Initialization

1. Clone the repository:

```
git clone xxx
cd MS-TFAL
```

2. Download the pre-trained reset model from [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) and place it in ```net/Ours/```

### Data Preparation

1. You need to download the [EndoVis18 Dataset](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Home/) and we recommend your directory tree be like:
```
$ MS-TFAL/dataset/
├── endovis18
│   ├── train
│   │   ├── seq_1
│       └── ├── labels
│           ├── left_frames
│       ├── seq_2
│           ...
│       └── seq_16
│
│   └── test
│       ├── seq_1
│       └── ├── labels
│           ├── left_frames
│       ├── seq_2
│           ...
│       └── seq_4
```

2. Generate the dataset with synthetic noise. See ```generate_noise/summary.txt``` for details.

* Your final directory tree of the dataset with synthetic noise should be like this:

```
$ MS-TFAL/dataset/
├── endovis18
│   ├── train_noisy_label
│   │   ├── noisy_scene_labels_final_mask_v0
│       │   ├── seq_1
│           ├── seq_2
│               ...
│           └── seq_16 
```

* Some description of files in ```dataset/```:


```
Endovis2018_MS_TFAL.py                 ----> Dataloader for our proposed MS-TFAL with Endovis18 Dataset
Endovis2018_backbone.py                ----> Dataloader for backbone model and other baselines with Endovis18 Dataset
Colon_OCT.py                           ----> Dataloader for Rat Colon Dataset
```
* Make sure ```./dataset/Endovis2018_MS_TFAL.py``` works before next step. 

### Training Process
* Use ```python train_MS_TFAL.py``` to start training; an example parameter setting is like:
```
python train_MS_TFAL.py --dataset endovis2018 --arch puredeeplab18 --log_name MS_TFAL_noisyver_0 --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 0 --iterations 56000 --data_type noisy --data_ver 0 --h 256 --w 320 --T1 4 --T2 6 --T3 8
```

* More example parameter setting refer to ```exp.sh```.

* Some description of files:

```
train_MS_TFAL.py                       ----> train the proposed MS-TFAL
train_backbone.py                      ----> train backbone model and other baselines
test.py                                ----> test the trained model
TFAL_visualization.py                  ----> Generate affinity confidence and samples visualization for TFAL module
```

### Evaluation & Visualization

* Use ```python test.py``` to start evaluation; an example parameter setting is like:
```
python test.py --arch puredeeplab18 --log_name MS_TFAL_noisyver_0_ver_0 --t 1 --gpu 0 --checkpoint 1 --h 256 --w 320
```
* More example parameter setting refer to ```exp.sh```.
* You can also use ```python TFAL_visualization.py``` to:
1. Compute feature based affinity confidence for each video.
2. Generate sample figures related to temporal affinity.



## Acknowledgment
Part codes are adopted from [STswinCL](https://github.com/YuemingJin/STswinCL)
