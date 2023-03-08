# Multi-Scale Supervision with Temporal Feature Affinity Learning (MSS-TFAL)

## Introduction
* This is the implementation for our MSS-TFAL.

## Usage

### Requirements

### Preprocessing & Data preparation

1. Clone the repository:

```
git clone https://github.com/BeileiCui/MSS-TFAL.git
```

2. You need to download the [EndoVis18 Dataset](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Home/) and generate the synthetic noisy dataset. Make sure ./dataset/Endovis2018_MSS_TFAL.py works. We recommend your directory tree be like:
```
$MSSTFAL/dataset/
├── endovis18
│   ├── train clean
│   │   ├── seq 1
        └── ├── labels
            ├── left_frames
│   │   ├── seq 2
│           ...
│   │   └── seq 16
│   └── test clean
│       ├── seq 1
        └── ├── labels
            ├── left_frames
│       ├── seq 2
│           ...
│       └── seq 4
```

### Traing process
