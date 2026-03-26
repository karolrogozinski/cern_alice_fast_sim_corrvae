# Fast Simulations of the ZDC Calorimeter in ALICE (CERN) using Deep Generative Models

![Status](https://img.shields.io/badge/Status-Published-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![Domain](https://img.shields.io/badge/Domain-High%20Energy%20Physics-purple)

> **TL;DR:** This repository provides a refactored and heavily modified implementation of the `CorrVAE` architecture, specifically adjusted for accelerating Zero Degree Calorimeter (ZDC) simulations in the **ALICE experiment at CERN**. It enables high-fidelity generation of calorimeter responses with explicit control over physical properties in the latent space.

## Project Overview

Traditional Monte Carlo simulations (like Geant4) used in high-energy physics are computationally expensive. This project introduces a deep learning-based fast simulation framework that generates ZDC calorimeter responses orders of magnitude faster while maintaining rigorous control over the generated data's physical properties.

**Read our full publication for detailed methodology and physical validation:**
📄 [Fast simulations of the ZDC calorimeter in ALICE CERN using deep machine learning with control over the properties of generated data](https://arxiv.org/abs/2405.14049)

## Dataset & Physical Properties

The dataset is derived from ZDC Fast Simulations within the ALICE framework. It comprises **295,867 responses**, each shaped as a `(1, 44, 44)` tensor representing calorimeter energy deposits.

To ensure physical validity, the generative model is conditioned on key spatial and energy properties (normalized to 0-1 range):
* X and Y coordinates of the maximum energy pixel
* X and Y coordinates of the mass center (energy distribution center)
* Number of non-zero pixels (shower extent)
* Categorized number of non-zero pixels (5 distinct spatial bins)
* Number of pixels with energy >1 (post-logarithmic scaling)
* Total sum of pixels (total deposited energy)
* Maximum pixel value (peak energy)

## Experiments & Latent Space Control

Our modified `CorrVAE` allows for disentangled control over specific physical traits during generation.

### 1. Property-Conditioned Generation
By fixing specific target properties and sampling a random **Z** latent vector, the model generates diverse but physically constrained calorimeter showers.

**Target Properties:** `X mass center: 10` | `Y mass center: 30` | `Shower Size: 0.3`

![Generation Example 1](https://github.com/karolrogozinski/cern_alice_fast_sim_corrvae/assets/73389492/aad0e42b-ef6f-4ffc-9099-866d4c2fc149)

**Target Properties:** `X mass center: 20` | `Y mass center: 20` | `Shower Size: 0.9`

![Generation Example 2](https://github.com/karolrogozinski/cern_alice_fast_sim_corrvae/assets/73389492/2cbe96ca-055a-4d41-adc1-c99daba9cb65)

### 2. Traversing the **W** Latent Space
By traversing elements in the isolated **W** latent space, we can smoothly interpolate between different physical states (e.g., shifting the shower center or increasing total energy) without recalculating the entire collision physics.

![animation](https://github.com/karolrogozinski/cern_alice_fast_sim_corrvae/assets/73389492/244f1571-d704-47aa-a30c-03e8eafda6fc)

## Code Structure & Architecture

The codebase is structured for modular experimentation with Variational Autoencoders.

```text
├── data/
│   └── prepare_data.py       # Data transformation pipelines for ZDC tensors
├── evaluations/
│   └── eval.ipynb            # Empirical evaluation and latent space visualization
├── modeling/
│   ├── model.py              # Main VAE architecture definition
│   ├── encoders.py           # Feature extraction modules
│   ├── decoders.py           # Generative upsampling modules
│   └── optim.py              # Optimization routines (e.g., latent w recovery)
├── utils/
│   ├── loss.py               # Custom highly-constrained loss functions
│   ├── model_init.py         # Network weight initialization routines
│   ├── spectral_norm_fc.py   # Spectral normalization (Miyato et al.)
│   └── train_helpers.py      # Training loop utilities
└── train.ipynb               # Primary training execution environment
```

## Quickstart

The environment and dependencies can be recreated using standard Python tooling. 

*(Note: Full CLI execution is currently being wrapped; primary training execution is handled interactively via `train.ipynb` for Colab/Jupyter optimization).*

```bash
git clone https://github.com/karolrogozinski/cern_alice_fast_sim_corrvae.git
cd cern_alice_fast_sim_corrvae
```

## References

**[1]** S. Wang et al., *"Multi-objective Deep Data Generation with Correlated Property Control"*, [arXiv:2210.01796](https://arxiv.org/abs/2210.01796), 2022.  
**[2]** T. Miyato et al., *"Spectral Normalization for Generative Adversarial Networks"*, [arXiv:1802.05957](https://arxiv.org/abs/1802.05957), 2018.
