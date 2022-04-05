# Automated cancer diagnostics via analysis of optical and chemical images by deep and shallow learning


This is the repository for the paper "Automated cancer diagnostics via analysis of optical and chemical images by deep and shallow learning". 


## Deep Learning
### Pre-processing of training data (WSI)
For the pre-processing of the training data, the default pipeline presented in the paper "Giunchiglia, V., McKenzie, J., and Takats Z., "WSIQC: whole slide images’ pre-processing pipeline for quality control assessment and AI-based data analysis", in preparation, 2021" was used. The code is avaialble at this link https://github.com/valegiunchiglia/wsi_pre_processing.

### Pre-processing of test data (TMA FFPE)
The pre-processing of FFPE images was characterised by the following steps:
1. Splitting of FFPE into patches with one core each
