# CorrVAE for manipulating properties in CERN Alice Simulations

Refactored and modified version of CorrVAE [1], adjusted for handling images ffrom CERN Alice fast simmulations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Run](#run)
4. [Data](#data)
5. [Code Structure](#code-structure)
6. [References](#references)

## Project Overview

TBA

## Requirements

TODO

## Run

TODO

## Data

TODO

## Code Structure
```
├──  data  
│    └──  prepare_data.py     - code used to make all data transformations and create final dataset
│
│
├──  evaluations
│    └──  eval.ipynb          - notebook that contains experiments on trained model 
│
│
├──  modeling
│    └──  decoders.py         - just encoder
│    └──  encoders.py         - just decoder
│    └──  model.py            - main model file
│    └──  optim.py            - probably optimizing getting w from y (in progress!)
│
│
├──  utils
│    └──  loss.py             - all custom loss functions used in training
│    └──  model_init.py       - model weights initialization
│    └──  spectral_norm_fc.py - spectral norm implementation from [2]
│    └──  train_helpers.py    - helpers for training loop
│
│
└──  train.ipynb              - training loop
```

## References

**[1]** Shiyu Wang, Xiaojie Guo, Xuanyang Lin, Bo Pan, Yuanqi Du, Yinkai Wang, Yanfang Ye, Ashley Ann Petersen, Austin Leitgeb, Saleh AlKhalifa, Kevin Minbiole, William Wuest, Amarda Shehu, Liang Zhao.</br>
    *[Multi-objective Deep Data Generation with Correlated Property Control](https://arxiv.org/pdf/2210.01796)*, 2022.

**[2]** Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.</br>
    *[Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)*, 2018.
