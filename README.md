# Automated Cancer Diagnostics via Analysis of Optical and Chemical Images by Deep and Shallow Learning


This is the repository for the paper "Automated cancer diagnostics via analysis of optical and chemical images by deep and shallow learning". 


## Deep Learning
### Pre-processing of training data (WSI)
For the pre-processing of the training data, the default pipeline presented in the paper "Giunchiglia, V., McKenzie, J., and Takats Z., "WSIQC: whole slide images’ pre-processing pipeline for quality control assessment and AI-based data analysis", in preparation, 2022" was used. The code will be available at this link https://github.com/valegiunchiglia/wsi_pre_processing. 

### Pre-processing of test data (TMA FFPE)
The main pre-processing steps of FFPE images were:
1. Automated splitting of the TMA image into patches each containing one core
2. Detection and removal of the background from high resolution images
3. Tiling of high resolution images
4. Selection of the tiles containing less than a specific propotion of background
5. Preparation of the dictionaries to input into the neural network

## Basic usage
```
# First command pre-processing pipeline

python3 pre_processing_tma_ffpe.py \
  --path_to_slides [path/to/images] \
  --output [path/to/output/directory] \
  --output_h5 [path/to/output/directory/h5files]
 
Optional arguments:
--thresh_back                Upper threshold of proportion of background allowed in a tile
--tile_size                  Size of tiles extracted from the Whole Slide Image


# First command for the calculation of performance metrics

python3 confusion_matrix_ffpe.py \
  --path_to_predictions [path/to/output/predictions] \
  --test_dictionary [path/to/test/dictionary] \
  --output [path/to/output/directory] \
  --path_to_images [path/to/images]
 
```

## Input data 

The **test dictionary** must have the following keys:
- ```slides```: list of paths to the images
- ```grid```: list of list of tuples of (x, y) coordinates  
- ```targets```: list of targets (either 1 or 0)
- ```h5files```: list of paths to the h5 files (one for each image)

The **probability dictionary** must have the following keys:
- ```probs```: list of probabilities obtained as output of the ML model
- ```slideIDX```: list of indices that identify to which image each tile belongs (tiles from the same image will have the same index)
