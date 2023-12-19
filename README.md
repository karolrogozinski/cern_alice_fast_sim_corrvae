# CorrVAE for manipulating properties in CERN Alice Simulations

Refactored and modified version of CorrVAE [1], adjusted for handling images ffrom CERN Alice fast simmulations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Run](#run)
4. [Data](#data)
5. [Code Structure](#code-structure)
6. [Experiments](#experiments)
7. [References](#references)

## Project Overview

TBA

## Requirements

TBA

## Run

Running project from console is not ready yet.
Currently used form of training is running train.ipynb, using Google Colab.

## Data

Dataset used for model and experiments is created from ZDC Fast Simulation in Alice CERN project. It contains 295867 responses in (1, 44, 44) shape and value bigger than 1.

Properties created for experiments:
 - X coordinate of max pixel
 - Y coordinate of max pixel
 - X coordinate of mass center
 - Y coordinate of mass center
 - Number of non-zero pixels
 - Categorized number of non-zero pixels (5 distinct values)
 - Number of pixels bigger than 1 (after applying log)
 - Sum of pixels
 - Max pixel value

All of them are scaled to 0-1 range.

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

## Experiments

Some examples of trained model:

### Generating images with given properites and random z latent space:

 1. Properties:
    - X mass center: 10
    - Y mass center: 30
    - size: 0.3
</br>

<img width="1020" alt="Screenshot 2023-12-06 at 2 56 23 PM" src="https://github.com/karolrogozinski/cern_alice_fast_sim_corrvae/assets/73389492/aad0e42b-ef6f-4ffc-9099-866d4c2fc149">

</br>
</br>

 2. Properties:
    - X mass center: 20
    - Y mass center: 20
    - size: 0.9
</br>

<img width="1018" alt="Screenshot 2023-12-06 at 3 05 01 PM" src="https://github.com/karolrogozinski/cern_alice_fast_sim_corrvae/assets/73389492/2cbe96ca-055a-4d41-adc1-c99daba9cb65">

### Traversing w latent space

Changing properties of responses by traversing elements in latent space, coresponding with given properties:
</br>

![animation](https://github.com/karolrogozinski/cern_alice_fast_sim_corrvae/assets/73389492/244f1571-d704-47aa-a30c-03e8eafda6fc)

</br>

## References

**[1]** Shiyu Wang, Xiaojie Guo, Xuanyang Lin, Bo Pan, Yuanqi Du, Yinkai Wang, Yanfang Ye, Ashley Ann Petersen, Austin Leitgeb, Saleh AlKhalifa, Kevin Minbiole, William Wuest, Amarda Shehu, Liang Zhao.</br>
    *[Multi-objective Deep Data Generation with Correlated Property Control](https://arxiv.org/pdf/2210.01796)*, 2022.

**[2]** Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.</br>
    *[Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)*, 2018.
