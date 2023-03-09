# Multi-Scale Supervision with Temporal Feature Affinity Learning (MSS-TFAL)

## Introduction
* This is the implementation for our MSS-TFAL.

## Usage

### Requirements

* pytorch
* numpy
* sklearn
* skimage
* tqdm
* tensorboardX

### Initialization

1. Clone the repository:

```
git clone https://github.com/BeileiCui/MSS-TFAL.git
cd MSS-TFAL
```

2. Download the pretrained reset model from [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) and place it in ```./net/Ours/```

### Data preparation
1. You need to download the [EndoVis18 Dataset](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Home/) and we recommend your directory tree be like:
```
$./MSSTFAL/dataset/
├── endovis18
│   ├── train_clean
│   │   ├── seq_1
│       └── ├── labels
│           ├── left_frames
│       ├── seq_2
│           ...
│       └── seq_16
│
│   └── test_clean
│       ├── seq_1
│       └── ├── labels
│           ├── left_frames
│       ├── seq_2
│           ...
│       └── seq_4
```

2. Convert the RGB label to class-level label with:

```
python dataset/to_class_label.py
```

3. Generate the dataset with synthetic noise. See ```generate_noise/summary.txt``` for details.


* Some description of files in ```./dataset```:

```
│
│   ├── train_noisy_label
│   │   ├── noisy_scene_labels_final_mask_v0
│       │   ├── seq_1
│           └── ├── framexxx.png
│               ├── framexxx.png
│           ├── seq_2
│               ...
│           └── seq_16
│ 
```

```
Endovis2018_MSS_TFAL.py                ----> Dataloader for our proposed MSS-TFAL with Endovis18 Dataset
Endovis2018_backbone.py                ----> Dataloader for backbone model and other baselines with Endovis18 Dataset
Colon_OCT.py                           ----> Dataloader for Rat Colon Dataset
```
* Make sure ```./dataset/Endovis2018_MSS_TFAL.py``` works before next step. 

### Traing process
* Use ```python train_MSS_TFAL.py``` to start training; an example parameter setting is like:
```
python train_MSS_TFAL.py --dataset endovis2018 --arch puredeeplab18 --log_name MSS_TFAL_noisyver_3 --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 0 --iterations 56000 --data_type noisy --data_ver 4 --h 256 --w 320 --T1 4 --T2 6 --T3 8
```

* More example parameter setting refer to ```exp.sh```.

* Some description of files in ```./```:

```
train_MSS_TFAL.py                      ----> train the proposed MSS-TFAL
train_backbone.py                      ----> train backbone model and other baselines
test.py                                ----> test the trained model
TFAL_visualization.py                  ----> Generate affinity confidence and samples visualization for TFAL module
```

### Evaluation & Visualization

* Use ```python test.py``` to start evaluation; an example parameter setting is like:
```
python test.py --arch puredeeplab18 --log_name MSS_TFAL_noisyver_3_ver_0 --t 1 --gpu 0 --checkpoint 1 --h 256 --w 320
```
* More example parameter setting refer to ```exp.sh```.
