# Neurological Outcome Prediction After Cardiac Arrest: A Multi-Level  Deep Learning Approach with Feature and Decision Fusion

## Resources

* [Moody Challenge](https://moody-challenge.physionet.org/2023/)

## Directory Structure

The main components of the project pipeline includes: 
* Preprocessing and transfering the raw data
* Split data to train, test, and validation set
* Training models for classification

 Each of those components have their respective directory. All programs/codes related to a component should exist in their related directory or sub-directory. 

```bash
.
├── artifacts
├── classifier
│   └── sweep-images
├── cnn-feature-extraction
├── figures
└── notebooks

```

## Setup

1. Clone this repository.
2. Create a Conda environment from `.yaml` file.
```
conda env create --file environment.yaml
```
3. Ensure that you have a data directory in your current local workspace.
