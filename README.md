# Multi-Metric-Based Brain Functional-Structural Connectivity Mutual Coupling for Mild Cognitive Impairment Identification
[2025/12] The code will be fully published after inclusion. Thank you for your attention.



 
## Overview
Recently, an increasing number of studies on MCI identification have started to focus on multi-modal fusion. However, extracting effective coupled information between functional connectivity and structural connectivity from brain networks remains a formidable challenge. In terms of coupled information extraction, methods that utilize one modality to guide another modality in network construction have been demonstrated to be advantageous in learning network representations for specific brain disease identification. Nevertheless, existing guidance mechanisms are often unimodal-guided, neglecting the counteraction of the other modality, which results in the loss of more multi-modal coupled information. Meanwhile, for multi-modal data derived from different indicators, discrepancies frequently exist, yet most studies only focus on single multi-modal data and overlook the diversity among multi-modal data. This paper proposes a multi-index multi-modal fusion mechanism to integrate the differences among multi-modal data for analyzing brain networks to diagnose MCI. In addition, we also propose a patch masking module, which represents brain network connections via patches to capture local information of brain networks. Finally, we put forward a cross-modal cross-guidance strategy to explicitly integrate the coupled information between modalities. Experiments on the public dataset ADNI and the validation dataset START demonstrate that our model outperforms existing studies in multiple metrics, and the proposed multi-modal multi-index mechanism is of significance.


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
