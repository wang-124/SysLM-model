# SysLM
A Systematic Longitudinal Study of Microbiome: Integrating Temporal-Spatial Dimensions with Causal and Deep Learning Models

## Introduction
SysLM is a comprehensive framework designed to analyze longitudinal microbiome data, integrating deep learning models, causal inference, and biological interpretation. It enables missing value imputation, disease prediction, and multi-level biomarker discovery across time and disease conditions.  

## Installation

SysLM requires Python ≥ 3.8 and CUDA ≥ 11.0. It has been tested with:

- Python 3.8  
- CUDA 12.2  
- PyTorch 2.0+

Option 1: Install via pip

-pip install syslm

Option 2: Install from source

git clone https://github.com/yourusername/syslm.git
cd syslm
pip install -r requirements.txt



## Usage
Step 1: Run Imputation Example

-python imputation_example.py

This script demonstrates how to use the SysLM-Impute module to infer missing values in longitudinal microbiome data.

Step 2: Biomarker Analysis and Prediction

-python syslmc_biomarker_analysis_example.py

Demonstrates how to perform disease prediction using the SysLM-C model, and extract three causal interpretability spaces for downstream biomarker analysis.

## Biomarker Discovery Modules

1.Differential and Network Biomarkers:via syslm.differential_and_network_biomarkers.py. Network biomarkers require Gephi Louvain algorithm (default parameters) for community detection and visualization

2.Core Biomarkers: via syslm.core_biomarker.py

3.Dynamic Biomarkers: via syslm.dynamic_biomarker.py

4. Disease-Specific vs. Shared Biomarkers
After obtaining the candidate biomarkers from above modules, you can derive:

Disease-specific biomarkers: by computing the set difference of biomarker sets between diseases

Shared biomarkers: by computing the set intersection across diseases