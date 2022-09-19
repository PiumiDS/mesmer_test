from deepcell.utils.plot_utils import create_rgb_image
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import make_outline_overlay
from deepcell.datasets import multiplex_tissue
from skimage.io import imread
import cv2
import numpy as np
import tifffile
import os, sys
from glob import glob
import pandas as pd
import pickle



tile_resize = (256, 256)
tile_size 	= (512, 512)
channel_image_folder = '/srv/scratch/bic/piumi/TMA_exported_images/'
EXT = '.tif'
ch_list = [8,2,3,4,6,7,5]
app = Mesmer()

def generate_tiles(raw_image, channel_list, run_combat=False):
    out = {}
    for channel in channel_list:
        image = raw_image[channel]
        im_h, im_w = image.shape # 4188, 5592
        (r_, c_) = (int(im_h/tile_size[0]), int(im_w/tile_size[1]))
        extract_tiles = [image[h_i*tile_size[0]: (h_i+1)*tile_size[0], w_i*tile_size[1]: (w_i+1)*tile_size[1]] for h_i in range(r_) for w_i in range(c_)]
        tile_coordinates = [(h_i, w_i) for h_i in range(r_) for w_i in range(c_)]
        out[channel] = [extract_tiles, tile_coordinates, np.amax(image), np.amin(image)]

    return out

for sub_folder in os.listdir(channel_image_folder):
    sub_folder_path = os.path.join(channel_image_folder, sub_folder)
    slide_num = sub_folder[-1]
    if int(slide_num) not in [9] : continue
    file_prefix = '/EM TNBCa {}_Core*.tif'.format(slide_num)
    save_path_ = '/srv/scratch/bic/piumi/mesmer/slide{}/'.format(slide_num)
    print(sub_folder_path, file_prefix, '\n', save_path_)
    count = 0
    for im_file_path in glob(str(sub_folder_path) + file_prefix):
        core_base_name = os.path.basename(im_file_path)
        _, rN, cN = core_base_name.split(']_[')[0].split('[')[1].split(',')

        print('reading tiff', im_file_path)
        tif_image = tifffile.imread(im_file_path)
        print('finished reading...')
        # order of channels: PD-1 0, FoxP3 1, PDL1 2, CD20 3, PanCK 4, CD8 5, CD3 6, CD68 7, DAPI 8
        data = generate_tiles(tif_image, ch_list)
        
        X_data = []
        for idx in range(len(data[ch_list[0]][0])): # assume atleast 1 channel in channel list
            
            im1 = cv2.resize(data[8][0][idx], tile_resize, interpolation=cv2.INTER_AREA)
            im1 = ((im1 - data[8][3])*255/(data[8][2] - data[8][3])).astype(np.uint8)

            #threshold
            if np.amax(im1) < (255*0.10) and np.sum(im1) < (255*0.10*30*10) : continue

            im2 = cv2.resize(data[2][0][idx], tile_resize, interpolation=cv2.INTER_AREA)
            im2 = ((im2 - data[2][3])*255/(data[2][2] - data[2][3])).astype(np.uint8)

            im3 = cv2.resize(data[3][0][idx], tile_resize, interpolation=cv2.INTER_AREA)
            im3 = ((im3 - data[3][3])*255/(data[3][2] - data[3][3])).astype(np.uint8)

            im4 = cv2.resize(data[4][0][idx], tile_resize, interpolation=cv2.INTER_AREA)
            im4 = ((im4 - data[4][3])*255/(data[4][2] - data[4][3])).astype(np.uint8)
            
            im5 = cv2.resize(data[6][0][idx], tile_resize, interpolation=cv2.INTER_AREA)
            im5 = ((im5 - data[6][3])*255/(data[6][2] - data[6][3])).astype(np.uint8)
            
            im6 = cv2.resize(data[7][0][idx], tile_resize, interpolation=cv2.INTER_AREA)
            im6 = ((im6 - data[7][3])*255/(data[7][2] - data[7][3])).astype(np.uint8)

            im7 = cv2.resize(data[5][0][idx], tile_resize, interpolation=cv2.INTER_AREA)
            im7 = ((im7 - data[5][3])*255/(data[5][2] - data[5][3])).astype(np.uint8)

            im = np.stack((im1, im2, im3, im4, im5, im6, im7), axis=-1)
            X_data.append(im)
            
            count += 1

            # if count ==15: break

        X_data = np.array(X_data)
        # print(X_data.shape)
        PDL1    = np.stack((X_data[:,:,:,0],X_data[:,:,:,1]), axis=-1)
        CD20    = np.stack((X_data[:,:,:,0],X_data[:,:,:,2]), axis=-1)
        PaCK    = np.stack((X_data[:,:,:,0],X_data[:,:,:,3]), axis=-1)
        CD3     = np.stack((X_data[:,:,:,0],X_data[:,:,:,4]), axis=-1)
        CD68    = np.stack((X_data[:,:,:,0],X_data[:,:,:,5]), axis=-1)
        CD8     = np.stack((X_data[:,:,:,0],X_data[:,:,:,6]), axis=-1)
        
        # nuclear_image = app.predict(PaCK, image_mpp=0.5, compartment='nuclear')
        PDL1_image    = app.predict(PDL1, image_mpp=0.5, compartment='both')
        CD20_image    = app.predict(CD20, image_mpp=0.5, compartment='both')
        PaCK_image    = app.predict(PaCK, image_mpp=0.5, compartment='both')
        CD3_image     = app.predict(CD3,  image_mpp=0.5, compartment='both')
        CD68_image    = app.predict(CD68, image_mpp=0.5, compartment='both')
        CD8_image     = app.predict(CD8,  image_mpp=0.5, compartment='both')
        
        new_dict ={}
        new_dict['X_data'] = X_data
        new_dict['nuclear_pred'] = PaCK_image[:,:,:,0]
        new_dict['PDL1_pred'] = PDL1_image[:,:,:,1]
        new_dict['CD20_pred'] = CD20_image[:,:,:,1]
        new_dict['PaCK_pred'] = PaCK_image[:,:,:,1]
        new_dict['CD3_pred']  = CD3_image[:,:,:,1]
        new_dict['CD68_pred'] = CD68_image[:,:,:,1]
        new_dict['CD8_pred']  = CD8_image[:,:,:,1]

        base = 'slide_' + str(slide_num) + '_core_' + str(rN) + '_' + str(cN)
        pickle_file = os.path.join(save_path_, base +'.p')
        with open(pickle_file, 'wb') as fp:
            pickle.dump(new_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # # create rgb overlay of image data for visualization
        # rgb_images = create_rgb_image(X_data[:,:,:,:2], channel_colors=['green', 'blue'])
        # overlay_nuclear = make_outline_overlay(rgb_data=rgb_images, predictions=app.predict(PaCK, image_mpp=0.5, compartment='nuclear'))
        # overlay_cell = make_outline_overlay(rgb_data=rgb_images, predictions=app.predict(PaCK, image_mpp=0.5, compartment='whole-cell'))
        # for i in range(X_data.shape[0]):
        #     cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/slide1/'+'{}{}_rbg.tif'.format(i, core_base_name),rgb_images[i, :,:,:])
        #     cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/slide1/'+'{}{}_nuc.tif'.format(i, core_base_name),overlay_nuclear[i, :,:,:])
        #     cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/slide1/'+'{}{}_cell.tif'.format(i, core_base_name),overlay_cell[i, :,:,:])

