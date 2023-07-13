# [MICCAI'2023] Rectifying Noisy Labels with Sequential Prior: Multi-Scale Temporal Feature Affinity Learning for Robust Video Segmentation

![Image](https://github.com/BeileiCui/MS-TFAL/blob/main/main_figure.jpg)

[__arxiv__](https://arxiv.org/abs/2307.05898)
## Introduction
* This is the implementation for our __MS-TFAL__. Noisy label problems are inevitably in existence within medical image segmentation causing severe performance degradation. Previous segmentation methods for noisy label problems only utilize a single image while the potential of leveraging the correlation between images has been overlooked. Especially for video segmentation, adjacent frames contain rich contextual information beneficial in cognizing noisy labels. Based on two insights, we propose a Multi-Scale Temporal Feature Affinity Learning (MS-TFAL) framework to resolve noisy-labeled medical video segmentation issues. First, we argue the sequential prior of videos is an effective reference, i.e., pixel-level features from adjacent frames are close in distance for the same class and far in distance otherwise. Therefore, Temporal Feature Affinity Learning (TFAL) is devised to indicate possible noisy labels by evaluating the affinity between pixels in two adjacent frames. We also notice that the noise distribution exhibits considerable variations across video, image, and pixel levels. In this way, we introduce Multi-Scale Supervision (MSS) to supervise the network from three different perspectives by re-weighting and refining the samples. This design enables the network to concentrate on clean samples in a coarse-to-fine manner. Experiments with both synthetic and real-world label noise demonstrate that our method outperforms recent state-of-the-art robust segmentation approaches.

## Usage

### Requirements

* pytorch
* opencv
* sklearn
* skimage
* tqdm
* tensorboardX

### Initialization

1. Clone the repository:

```
git clone https://github.com/BeileiCui/MS-TFAL.git
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
Part codes are adopted from [STswinCL](https://github.com/YuemingJin/STswinCL).
