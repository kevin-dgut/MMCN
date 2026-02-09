# Multimetric Brain Functional-Structural Connectivity Mutual Coupling for Mild Cognitive Impairment Identification
[2026/02] This is the official PyTorch implementation of MMCN from the paper "Multimetric Brain Functional-Structural Connectivity Mutual Coupling for Mild Cognitive Impairment Identification".


 
## Overview
Mild cognitive impairment (MCI) is an intermediate stage between normal cognition and neurodegenerative conditions such as Alzheimer’s disease, where timely identification is crucial for effective intervention. MRI data-derived brain connectome analysis based on functional connectivity (FC) and structural connectivity (SC) is a promising strategy for MCI identification. However, the performances of existing approaches are limited by their reliance on single connectivity metrics and unidirectional SC-FC fusion method. In this work, we propose a multimetric mutual coupling network (MMCN) to model multimetric brain connectivities and bidirectional FC-SC interactions for MCI identification. Subject-specific FC and SC networks are first constructed based on multiple functional and structural metrics to mitigate single-metric bias and capture complementary topological features. Then, the model incorporates a specific FC/SC patch embedding module alongside a patch enhancement mechanism, which leverages cross-guided attention to enable mutual feature refinement between structural and functional modalities. This architecture explicitly models the mutual influence between neuronal activity and white matter integrity, characterizing the subtle yet critical connectivity alterations associated with MCI. The refined FC and SC features are finally fused into a mixed functional–structural  matrix and fed into a Transformer backbone for feature fusion learning and classification. We conduct extensive experiments on multi-modal MRI data from the public ADNI dataset and a local dataset cohort. The results demonstrate the superior performance of our approach and highlight the necessity of multimetric FC-SC multual coupling for MCI identification.

## Dateset
The public ADNI dataset used in the paper is downloaded from its official websites (https://adni.loni.usc.edu/).

## Requirements
Python 3.11.5
torch>=2.0.1
torchvision>=0.15.2
torchaudio>=2.0.2
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
tqdm>=4.64
Pillow>=9.4

## Optional
- GPU support is recommended for faster training
