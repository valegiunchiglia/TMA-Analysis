"""
Confusion Matrix for Formalin-Fixed-Paraffin-Embedded Tumor Microarray (FFPE-TMA)
samples for Deep-Learning based tumor prediction
Author: Valentina Giunchiglia
"""


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import openslide, torch, cv2, sys, os
from PIL import Image, ImageDraw
from tqdm import tqdm
from skimage import measure, filters, segmentation, color, morphology, feature, transform
import itertools as it
from scipy import signal, interpolate
import warnings
from tqdm import tqdm
import argparse




parser = argparse.ArgumentParser(description='Pipeline used to create confusion matrix of Deep Learning output')
parser.add_argument('--thresh_areas', nargs = "+", type=int, default='',
                    help='list of thresholds for the minimum number of positive pixels to classify a core as tumour')
parser.add_argument('--thresh_probs', , nargs = "+", type=int, default='',
                    help='list of thresholds on the output DL probability to classify a pixel as positive')
parser.add_argument('--output', type=str, default="", 
                    help='Output directory')
parser.add_argument('--prob_dict', type=str, default="", 
                    help='Path to the probability dictionary (output of DL model)')
parser.add_argument('--test_dict', type=str, default="", 
                    help='Path to test dictionary used to test the DL model')





def tumour_regions(prob_dict, train_dict, name_file, thresh_area, window_size, cores_keep, thresh_prob):
    # Find the probability that belong to the specific file
    indices = [e for e,name in enumerate(train_dict["slides"]) if name_file in name]
    
    # Create the heatmap
    mask_prob, image = heatmap_probs_tma(prob_dict, train_dict, indices)
    
    
    # Rescale the image for easier identification of the circles
    smaller_img = transform.rescale(np.array(image), 0.25, order=3, multichannel=True)
    
    # Use otsu to detect the circles
    gray_img = color.rgb2gray(smaller_img)
    img_otsu = gray_img < filters.threshold_otsu(gray_img)
    otsu_filtered = filters.median(img_otsu, morphology.disk(11)) > 0.5
    
    # Detect the number of connected regions
    tumor_region_stack, grid_regions, gridC, gridR = split_grid_3d_sorted(otsu_filtered, window_size)
    
    
    # Filter the regions with a probability below 0.5
    mask_prob2 = mask_prob.copy()
    mask_prob2[mask_prob2 >= thresh_prob] = 255
    lab, num = measure.label(mask_prob2, return_num=True)
    properties = ['area']
    df = pd.DataFrame(measure.regionprops_table(lab,mask_prob2, properties = properties))
    #df = df[(df['area'] > 500)]

    predicted_tum = mask_prob2.copy()
    
    # Resize the image with the predicted probabilities
    minitum = transform.resize(predicted_tum, tumor_region_stack.shape[1:], preserve_range=True)
    
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

    
    
    
def heatmap_probs_tma(prob_dict, train_dict, indices):
    
    slide = train_dict["slides"][indices[0]]
    img = openslide.OpenSlide(slide)
    img_thumb = img.get_thumbnail((4000, 4000))
    image_final = np.zeros(img_thumb.size[:2])
    scale_factor = img_thumb.size[0] / img.dimensions[0]
    slideID = prob_dict["slideIDX"]
    
    for k in tqdm(indices):
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

## Summarize as function
def split_as_grid(img, window_size=80):
    
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
    return [0]+rel_min_c+[img.shape[1]], [0]+rel_min_r+[img.shape[0]]
    #return [0]+rel_min_c, [0]+rel_min_r


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


def split_grid_3d_sorted(img, window_size):
    """
    Input is otsu image

    gridC, window = 0,0
    while gridC != ncols and window <= 2000:
        window += 10
        gridC, gridR = split_as_grid(img, window_size=window)
        
    if window > 200:
        warnings.warn("Optimal window size not found, {} columns produced".format(len(gridC)-2))
    """ 
    grid_c, grid_r = split_as_grid(img, window_size=window_size)
    
    grid_regions = define_squares_from_grid(grid_c, grid_r)
    stack = np.zeros((len(grid_regions), img.shape[0], img.shape[1]))
    stack[:, :, :] = np.expand_dims(img, 0)
    
    for z,((c,r),(w,h)) in enumerate(grid_regions):
        tmp = np.ones_like(img, dtype=bool)
        tmp[r:r+h, c:c+w] = False
        stack[z, tmp] = 0

    return stack, grid_regions, grid_c, grid_r



imgs = ["TMA52", "TMA53", "TMA51", "TMA58", "TMA54", "TMA63", "TMA55", "TMA59", "TMA60"]
img_healthy = ["TMA44", "TMA45"]



def main():
    global args
    args = parser.parse_args()
    
    prediction_performance = []
    for thresh in args.thresh_areas:
        for thresh_prob in args.thresh_probs:
            total_tumour = 0
            total_tumo_detected = 0
            for img in imgs:
                window_size = windows[img]
                cores_keep = areas_keep[img]
                region_present, N = tumour_regions(args.prob_dict, args.test_dict, img, thresh, window_size, cores_keep, thresh_prob)
                total_tumour += N
                total_tumo_detected += region_present

            total_he = 0
            he_detected = 0
            for img in img_healthy:
                window_size = windows[img]
                cores_keep = areas_keep[img]
                region_present, N = tumour_regions(args.prob_dict, args.test_dict, img, thresh, window_size, cores_keep, thresh_prob)
                total_he += N
                he_detected  += (N-region_present)

            perf = {
                "thresh_area":thresh, "thresh_prob":thresh_prob,
                "tp": total_tumo_detected, "tn": he_detected,
                "fp":total_he-he_detected, "fn":total_tumour-total_tumo_detected
            }

            prediction_performance.append(perf)


    pd.DataFrame(prediction_performance).to_csv("/rds/general/user/vg816/home/Zoltan_Data/TMA_prediction_performance_vs_area_threshold_complete.csv")

