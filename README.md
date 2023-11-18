# CorrVAE for manipulating properties in CERN Alice Simulations

Refactored and modified version of CorrVAE [1], adjusted for handling images ffrom CERN Alice fast simmulations.

## Project Overview

TBA

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
│    └──  model_init.py       - model initialization from [2]
│    └──  spectral_norm_fc.py - spectral norm implementation from [3]
│    └──  train_helpers.py    - helpers for training loop
│
│
└──  train.ipynb              - training loop
```
