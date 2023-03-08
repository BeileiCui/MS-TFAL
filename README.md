# Multi-Scale Supervision with Temporal Feature Affinity Learning (MSS-TFAL)

## Introduction
* This is the implementation for our MSS-TFAL.

## Usage

### Requirements

### Preprocessing & Data preparation

1. Clone the repository:

```
git clone https://github.com/BeileiCui/MSS-TFAL.git
cd MSS-TFAL
```

2. You need to download the [EndoVis18 Dataset](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Home/) and generate the dataset with synthetic noise. We recommend your directory tree be like:
```
$./MSSTFAL/dataset/
├── endovis18
│   ├── train clean
│   │   ├── seq 1
│       └── ├── labels
│           ├── left_frames
│       ├── seq 2
│           ...
│       └── seq 16
│
│   └── test clean
│       ├── seq 1
│       └── ├── labels
│           ├── left_frames
│       ├── seq 2
│           ...
│       └── seq 4
```

Some description of files in ```./dataset```:

```
Endovis2018_MSS_TFAL.py                ----> Dataloader for our proposed MSS-TFAL with Endovis18 Dataset
Endovis2018_backbone.py                ----> Dataloader for backbone model and other baselines with Endovis18 Dataset
Colon_OCT.py                           ----> Dataloader for Rat Colon Dataset
```
Make sure ```./dataset/Endovis2018_MSS_TFAL.py``` works before next step. 

### Traing process
Use ```python train_MSS_TFAL.py``` to start training; parameter setting refer to ```exo.sh```
