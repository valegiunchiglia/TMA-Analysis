"""
Confusion Matrix for Formalin-Fixed-Paraffin-Embedded Tumor Microarray (FFPE-TMA)
samples for Deep-Learning based tumor prediction
Author: Valentina Giunchiglia
"""
#!/Users/valentinagiunchiglia/anaconda3/bin/python

import os
import torch
import itertools as it
import numpy as np
import pandas as pd
import openslide
from tqdm import tqdm
from skimage import filters, color, morphology, transform
from scipy import signal
from constants import areas_keep, img_names_healthy, img_names_tumor, thresh_combis, windows
from datetime import datetime
import argparse

try:
  from tma_coords import tma_coords
except ModuleNotFoundError:
  pass


parser = argparse.ArgumentParser(description='TMA FFPE Performance evaluation')
parser.add_argument('--path_to_predictions', type=str, default='',
                    help='path to predictions')
parser.add_argument('--test_dictionary', type=str, default="",
                    help='Output directory for h5 files')
parser.add_argument('--output', type=str, default="", 
                    help='Output directory')
parser.add_argument('--path_to_images', type=str, default="", 
                    help='path to the folder containing the TMA images')




def main():
    global args
    args = parser.parse_args()
    prob_dict = torch.load(args.path_to_predictions)
    train_dict = torch.load(args.test_dictionary)

    prediction_performance = []
    for thresh_area, thresh_prob in tqdm(thresh_combis,
                                         desc = "Area/probs threshold combis"):
        total_tumour = 0
        total_tumo_detected = 0
        for file_name in img_names_tumor:
            window_size = windows[file_name]
            cores_keep = areas_keep[file_name]
            grid_regions = tma_coords[file_name]
            region_present, N = tumour_regions(
                prob_dict, train_dict, file_name, thresh_area, 
                window_size, cores_keep, thresh_prob, grid_regions, 
                args.path_to_images 
            )
            total_tumour += N
            total_tumo_detected += region_present

        total_he = 0
        he_detected = 0
        for file_name in img_names_healthy:
            window_size = windows[file_name]
            cores_keep = areas_keep[file_name]
            grid_regions = tma_coords[file_name]
            region_present, N = tumour_regions(
                prob_dict, train_dict, file_name, thresh_area, 
                window_size, cores_keep, thresh_prob, grid_regions, 
                args.path_to_images 
            )
            total_he += N
            he_detected  += (N-region_present)
    
        perf = {
            "thresh_area":thresh_area, "thresh_prob":thresh_prob,
            "tp": total_tumo_detected, "tn": he_detected,
            "fp":total_he-he_detected, "fn":total_tumour-total_tumo_detected
        }
    
        prediction_performance.append(perf)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pd.DataFrame(prediction_performance).to_csv(os.path.join(args.output, 
        f"TMA_prediction_performance_vs_area_threshold_complete_{now}.csv"
    ))


def tumour_regions(prob_dict, train_dict, file_name, thresh_area, window_size,
                    cores_keep, thresh_prob, grid_regions, path_to_images):
    # Find the probability that belong to the specific file
    indices = [e for e,name in enumerate(train_dict["slides"]) if file_name in name]
    
    # Create the heatmap
    mask_prob, image = heatmap_probs_tma(prob_dict, train_dict, indices, path_to_images)
    # Filter the regions with a probability below 0.5
    mask_prob2 = mask_prob.copy()
    mask_prob2[mask_prob2 >= thresh_prob] = 255
    predicted_tum = mask_prob2.copy()

    # Rescale the image for easier identification of the circles
    smaller_img = transform.rescale(np.array(image), 0.25, order=3, multichannel=True)
    # Use otsu to detect the circles
    gray_img = color.rgb2gray(smaller_img)
    img_otsu = gray_img < filters.threshold_otsu(gray_img)
    otsu_filtered = filters.median(img_otsu, morphology.disk(11)) > 0.5
    
    # Detect the number of connected regions
    tumor_region_stack, grid_regions = split_grid_3d_sorted(otsu_filtered,
                                                            window_size,
                                                            grid_regions,
                                                            whole_square = True)
    
    # Resize the image with the predicted probabilities
    minitum = transform.resize(predicted_tum, tumor_region_stack.shape[1:], 
    preserve_range=True)
    
    # Intersect the images to check which image is detected
    intersection = np.logical_and(
        tumor_region_stack,
        np.expand_dims(minitum.astype(int), 0)
    )
    
    # Classify as detected only if intersection has more than 500 pixels
    intersection_area = np.sum(intersection, axis=(1,2))
    areas_keep = intersection_area[cores_keep]
    total_regions = len(areas_keep)
    region_present = (areas_keep > thresh_area).sum()
    return region_present, total_regions
      

def heatmap_probs_tma(prob_dict, train_dict, indices, path_to_images):
    """
    Given the dictionary containing the output probabilities of the neural network
    and the dictionary used to test the performance of the model, it returns a 
    heatmap of the original TMA with the pixels detected as tumour highlightes
    """    
    slide = train_dict["slides"][indices[0]]
    print(slide)
    basename = os.path.basename(slide)
    img = openslide.OpenSlide(os.path.join(path_to_images, basename))
    img_thumb = img.get_thumbnail((4000, 4000))
    image_final = np.zeros(img_thumb.size[:2])
    scale_factor = img_thumb.size[0] / img.dimensions[0]
    slideID = prob_dict["slideIDX"]
    
    for k in indices:
        if k not in slideID:
            continue
        hres_coord = train_dict["grid"][k]
        lowres_coords = []
        for x, y in hres_coord:
            c = (round(int(np.multiply(scale_factor, x))), round(int(np.multiply(scale_factor, y))))
            lowres_coords.append(c)
        try:
            problist = prob_dict["probs"][slideID.index(k):slideID.index(k+1)]
        except ValueError:
            if k == max(slideID):
                problist = prob_dict["probs"][slideID.index(k):]
            else:
                next_k = [i for i in range(k+1, max(slideID)+1) if i in slideID]# [0]
                if next_k is None:
                    continue
                else:
                    problist = prob_dict["probs"][slideID.index(k):slideID.index(next_k[0])]
            
        tilesize_lr = int(scale_factor*224)
        
        for i, (x, y) in enumerate(lowres_coords):
            image_final[x:x+tilesize_lr,y:y+tilesize_lr] =  problist[i]
        
    return image_final.T, img_thumb


def split_as_grid(img, window_size=80):
    """
    Given an array, it returns the grid points on the rows and the columns
    """
    
    if len(img.shape) == 3:
        if img.shape[-1] == 3:
            img = color.rgb2gray(img)
        elif img.shape[-1] == 4:
            img = color.rgb2gray(color.rgba2rgb(img))

    win = signal.windows.hann(window_size)
    
    col_sums, row_sums = img.sum(0), img.sum(1)
    smooth_c = signal.convolve(col_sums, win, mode='same') / sum(win)
    smooth_r = signal.convolve(row_sums, win, mode='same') / sum(win)

    rel_min_c = signal.argrelmin(smooth_c)[0].tolist()
    rel_min_r = signal.argrelmin(smooth_r)[0].tolist()
    #return [0]+rel_min_c, [0]+rel_min_r
    return [0]+rel_min_c+[img.shape[1]], [0]+rel_min_r+[img.shape[0]]


def define_squares_from_grid(grid_cols, grid_rows):
    """
    Given the row and column where the grid should go, return the square
    coordinates (topleft pixels) and the width and height of every square
    """
    
    square_origin_points = it.product(grid_cols[:-1], grid_rows[:-1])
    square_widths = np.diff(grid_cols)
    square_heights = np.diff(grid_rows)
    square_dims = it.product(square_widths, square_heights )
    
    squares_info = list(zip(square_origin_points, square_dims))
    return squares_info


def split_grid_3d_sorted(otsu_img:np.array, window_size:int,
                         grid_regions:list=None, whole_square:bool=False):
    """
    Parameters
    ----------
    otsu_img      - otsu-filtered (binary) image representing a 
                    segmentation map of where the cores are
    window_size   - width of the window used for smoothing during grid estimation.
                    parameter passed to `split_as_grid`
    grid_regions  - list of tuples containing square coordinates of the form
                    ((topleft_x, topleft_y), (width, height))
    whole_square  - boolean: if False, each layer of the returned stack will be
                    True only where the core is located. Alternatively, the entire
                    square will be True
    """
    if grid_regions is None:
        grid_c, grid_r = split_as_grid(otsu_img, window_size=window_size)
        grid_regions = define_squares_from_grid(grid_c, grid_r)
    stack = np.zeros((len(grid_regions), otsu_img.shape[0], otsu_img.shape[1]))
    for z,((c,r),(w,h)) in enumerate(grid_regions):
        region_mask = np.zeros_like(otsu_img, dtype=bool)
        region_mask[r:r+h, c:c+w] = True
        if whole_square:
            stack[z, region_mask] = 1
        else:
            stack[z, region_mask] = otsu_img[region_mask]
    return stack, grid_regions


if __name__ == "__main__":
    main()
