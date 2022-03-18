import openslide, cv2, os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, morphology, filters, color
from scipy import signal, interpolate # Smooth the rowsums of the image when making grid
import itertools as it
from tqdm import tqdm
from openslide.deepzoom import DeepZoomGenerator


path = "/rds/general/user/vg816/projects/ambientms/ephemeral/breast_tma/"
h5_output_dir = "/rds/general/user/vg816/projects/ambientms/ephemeral/breast_tma_h5/"
path_save = "/rds/general/user/vg816/home/Zoltan_Data/"

def homogeneous_background(img, nconds=4):
    bckgmask = np.all(img == [[[0,0,0]]], axis=(2))
    img[bckgmask] = 255
    rgblist = img.reshape(-1, 3)

    r = np.multiply([12], rgblist[:,0])
    g = np.multiply(-10,rgblist[:,1])
    b = np.multiply(-1,rgblist[:,2])

    cond1 = -363 +r+g+b < 0
    cond2 = 100 +r+g+b > 0
    if nconds == 4:
        r = np.multiply(0.5,rgblist[:,0]) 
        g1 = np.multiply(0.5,rgblist[:,1])
        g2 = np.multiply(0.41,rgblist[:,1])
        cond3 = 35 + r + g2 + b > 0
        cond4 = -15 + r + g + b < 0
        back_px = np.logical_and.reduce((cond1, cond2, cond3, cond4))
    elif nconds==2:
        back_px = np.logical_and.reduce((cond1, cond2))
    
    return back_px.reshape(img.shape[0],img.shape[1])
    
    
def save_to_hdf5(db_location, tiles, grid, file_name):
    """
    Given the output directory, the tiles array, the coordinate information  and the name of the file,
    a hd5f file is created for each WSI, containing the information about each tiles
    as numpy array and the coordinate information.
    Parameters
    ----
    db_location(str) = directory where to save the files
    tiles(array) = list of tiles as numpy array
    grid(array) = list of coordinates of each tile
    file_name(str) = name of the file
    
    """
    # Save patches into hdf5 file.
    with h5py.File(db_location + file_name,'w') as file:
        dataset = file.create_dataset('tiles', np.shape(tiles), h5py.h5t.STD_I32BE, data=tiles)
        dataset = file.create_dataset('grid', (len(grid),2), h5py.h5t.STD_I32BE, data=grid)
        
        
        
def get_info(filename, squares_info, h5_output_dir, sf3):
    image = openslide.OpenSlide(filename)
    all_tiles = []
    hres_coords_list = []
    h5_names = []
    targets = []
    slides = []
    deepzoom = DeepZoomGenerator(image, tile_size=224, overlap=22, limit_bounds = True)
    for numm, ((x_th,y_th), (wi, he)) in enumerate(tqdm(squares_info)):
        (x0,y0) = (int(x_th*sf3),int(y_th*sf3)) # Convert thumbnail coordinates to zoom lvl 0 coords
        (wi, he) = (int(wi*sf3),int(he*sf3))

        this_tiles = []
        this_coords = []
        high_resolution_level = deepzoom.level_count - 1
        for x in range(deepzoom.level_tiles[-1][0]):

            ## Make sure that the slide is within the grid region
            tile_x, tile_y = deepzoom.get_tile_coordinates(high_resolution_level,(x,0))[0]
            if tile_x+224 > x0+wi:
                break
            for y in range (deepzoom.level_tiles[-1][1]):
                tile_x, tile_y = deepzoom.get_tile_coordinates(high_resolution_level,(x,y))[0]
                if  tile_x < x0 or (tile_y+224 > y0+he):
                    break
                if tile_y < y0:
                    continue

                tile = deepzoom.get_tile(high_resolution_level,(x,y))
                tile_array = np.array(tile)

                ### Background homogenization
                mask_background = homogeneous_background(tile_array)
                tile_array[mask_background] = 0

                #select only the tiles that have a percentage of background lower than the threshold
                perc_bckg = np.sum(tile_array == 0, axis = None) / tile_array.size
                if perc_bckg < 0.4:
                    this_tiles.append(np.array(tile))
                    this_coords.append((tile_x, tile_y)) 

        print("Number of tiles:", len(this_tiles))
        #extract the basename and create the h5 file
        basename = os.path.basename(filename)
        
        # Extract targets
        if "cancer" in basename:
            targets.append(1)
        elif "normal" in basename:
            targets.append(0)
        basename = basename[:basename.index(".")]
    
        h5_name = basename + "{:3d}.h5".format(numm)
        save_to_hdf5(h5_output_dir, this_tiles, this_coords, h5_name)

        #define the information that you need to run the deep learning model
        h5_names.append(h5_name)    
        all_tiles.append(this_tiles)
        hres_coords_list.append(this_coords)
        slides.append(filename)
        
    return all_tiles, hres_coords_list, h5_names, targets, slides



def split_tissue_as_grid(thumbnail, channel = "s"):
    """
    Given a thumbnail PIL image return the grid points on the rows and the columns
    """
    chann = {"h":0, "s":1, "v":2}
    thumb_arr = np.array(thumbnail)
    hsv = color.rgb2hsv(thumb_arr)[:,:,chann[channel]]
    
    rowsums, colsums = hsv.sum(1), hsv.sum(0)
    win = signal.windows.hann(300)
    
    smooth_rows = signal.convolve(rowsums, win, mode='same') / sum(win)
    rel_min_r = signal.argrelmin(smooth_rows)[0]
    rel_min_r = [0] + rel_min_r.tolist() + [hsv.shape[0]]

    smooth_cols = signal.convolve(colsums, win, mode='same') / sum(win)
    rel_min_c = signal.argrelmin(smooth_cols)[0]
    rel_min_c = [0] + rel_min_c.tolist() + [hsv.shape[1]]

    return rel_min_r, rel_min_c


def define_squares_from_grid(grid_rows, grid_cols):
    """
    Given the row and column where the grid should go, return the square
    coordinates (topleft pixels) and the width and height of every square
    """
    
    square_origin_points = it.product(grid_cols[:-1], grid_rows[:-1]) #Image edges are not square origins
    square_widths = np.diff(grid_cols)
    square_heights = np.diff(grid_rows)
    square_dims = it.product(square_widths, square_heights )
    
    squares_info = list(zip(square_origin_points, square_dims))
    return squares_info



def paths_to_slides(list_paths):
    """
    Given the different paths to the directories containing normal and tumor WSI, it creates a list of paths
    Parameters
    -----
    list_paths(array) = a array of paths (strings)
    """
    all_files = []
    for path in list_paths:
        filesinfolder = [path + file for file in os.listdir(path)]
        all_files.extend(filesinfolder)
    return all_files



def create_dictionary(grid, slides, hdf5, target, mult, level, save):
    """
    Given the information about the grid, slides, hdf5, required to train the MIL model, a dictionary
    that stores all the information is created. If save is True, then the dictionary is saved in the 
    specified directory. 
    Parameters
    ----
    grid(list of lists) = list of the coordinates for each WSI 
    slides(list) = list of paths to WSI (output of paths_to_slides)
    hdf5(list)  = list of paths to h5 files 
    save(str) = path where to save the dictionary. If None, no dictionary is saved
    """
    dict = {}   
    dict["grid"] = grid
    dict["slides"] = slides
    dict["targets"] = target
    dict["mult"] = mult
    dict["level"] = level
    dict["h5files"] = hdf5
    if save:
        torch.save(dict, os.path.join(save, "Testing_04.pth"))
        
        
def main(): 
    list_dir = [path]
    images_filename = paths_to_slides(list_dir)
    
    grid_final = []
    slides_final = []
    hdf5_final = []
    targets_final = []
    for fname in images_filename:
        basename = os.path.basename(fname)
        if not basename.startswith("."):
            image = openslide.OpenSlide(fname)
            thumb = image.get_thumbnail((4000, 4000))
            grows, gcols = split_tissue_as_grid(thumb)
            squares_information = define_squares_from_grid(grows, gcols)
        
            sf1 = image.level_dimensions[1][0]/thumb.size[0] # Level 1 / Thumbnail
            sf2 = image.level_dimensions[1][1]/image.dimensions[1] # Level 1 / Level 0
            sf3 = image.dimensions[0]/thumb.size[0] # Level 0 / Thumbnail
            tiles, coord, h5_name, targets, slides = get_info(fname, squares_information, h5_output_dir, sf3)
        
            h5_files = [h5_output_dir + h5 for h5 in h5_name]
        
            grid_final.extend(coord)
            hdf5_final.extend(h5_files)
            targets_final.extend(targets)
            slides_final.extend(slides)
        
    create_dictionary(grid = grid_final,slides = slides_final, target = targets_final, mult = 1, level = 0,hdf5 = hdf5_final, save = path_save)
            

if __name__ == "__main__":
    main()
