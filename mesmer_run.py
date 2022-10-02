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

#this code assumes atleast 1 channel in channel list and DAPI (channel 8) as an essential channel for quality testing

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
        im_h, im_w = image.shape # (4188, 5592)
        (r_, c_) = (int(im_h/tile_size[0]), int(im_w/tile_size[1]))
        intensity_check = (np.percentile(image, 99.95) < np.amax(image)/5)
        if intensity_check: 
            image = np.zeros((im_h, im_w))
            image_norm = np.zeros((im_h, im_w))
        else:
            image_norm = (image - np.amin(image))*255/(np.amax(image) - np.amin(image))
        
        extract_tiles = [image_norm[h_i*tile_size[0]: (h_i+1)*tile_size[0], w_i*tile_size[1]: (w_i+1)*tile_size[1]] for h_i in range(r_) for w_i in range(c_)]
        tile_coordinates = [(h_i, w_i) for h_i in range(r_) for w_i in range(c_)]
        out[channel] = [extract_tiles, tile_coordinates]

    return out

for sub_folder in os.listdir(channel_image_folder):
    sub_folder_path = os.path.join(channel_image_folder, sub_folder)
    slide_num = int(sub_folder[-1])
    # if slide_num in [1,2,3,4,5] : continue
    file_prefix = '/EM TNBCa {}_Core*.tif'.format(slide_num)
    save_path_ = '/srv/scratch/bic/piumi/mesmer/all_slides_new/slide{}/'.format(slide_num)
    print(os.path.basename(sub_folder_path), '#####', file_prefix, '#####', save_path_)
    count = 0
    for im_file_path in glob(str(sub_folder_path) + file_prefix):
        core_base_name = os.path.basename(im_file_path)
        _, rN, cN = core_base_name.split(']_[')[0].split('[')[1].split(',')

        print('reading tiff...', os.path.basename(im_file_path))
        tif_image = tifffile.imread(im_file_path)
        print('finished reading')
        # check DAPI quality
        DAPI_check = np.percentile(tif_image[8], 99.95) < (np.amax(tif_image[8])/5)
        if DAPI_check : 
            print('removed core')
            continue
        # order of channels: PD-1 0, FoxP3 1, PDL1 2, CD20 3, PanCK 4, CD8 5, CD3 6, CD68 7, DAPI 8
        data = generate_tiles(tif_image, ch_list)
        
        X_data = []
        for idx in range(len(data[ch_list[0]][0])): # assume atleast 1 channel in channel list
            
            im1 = cv2.resize(data[8][0][idx], tile_resize, interpolation=cv2.INTER_AREA).astype(np.uint8)

            #threshold
            if np.amax(im1) < (255*0.10) and np.sum(im1) < (255*0.10*30*10) : continue

            im2 = cv2.resize(data[2][0][idx], tile_resize, interpolation=cv2.INTER_AREA).astype(np.uint8)

            im3 = cv2.resize(data[3][0][idx], tile_resize, interpolation=cv2.INTER_AREA).astype(np.uint8)

            im4 = cv2.resize(data[4][0][idx], tile_resize, interpolation=cv2.INTER_AREA).astype(np.uint8)
            
            im5 = cv2.resize(data[6][0][idx], tile_resize, interpolation=cv2.INTER_AREA).astype(np.uint8)
            
            im6 = cv2.resize(data[7][0][idx], tile_resize, interpolation=cv2.INTER_AREA).astype(np.uint8)

            im7 = cv2.resize(data[5][0][idx], tile_resize, interpolation=cv2.INTER_AREA).astype(np.uint8)

            im = np.stack((im1, im2, im3, im4, im5, im6, im7), axis=-1)
            X_data.append(im)
            
            count += 1

            # if count ==15: break

        X_data = np.array(X_data)
        nuclear = np.stack((X_data[:,:,:,0],X_data[:,:,:,0]), axis=-1)
        PDL1    = np.stack((X_data[:,:,:,0],X_data[:,:,:,1]), axis=-1)
        CD20    = np.stack((X_data[:,:,:,0],X_data[:,:,:,2]), axis=-1)
        PaCK    = np.stack((X_data[:,:,:,0],X_data[:,:,:,3]), axis=-1)
        CD3     = np.stack((X_data[:,:,:,0],X_data[:,:,:,4]), axis=-1)
        CD68    = np.stack((X_data[:,:,:,0],X_data[:,:,:,5]), axis=-1)
        CD8     = np.stack((X_data[:,:,:,0],X_data[:,:,:,6]), axis=-1)

        X_data_array = np.zeros(X_data.shape)
        nuclear_array= np.zeros((X_data.shape[0], tile_resize[0], tile_resize[1]))
        PDL1_array   = np.zeros((X_data.shape[0], tile_resize[0], tile_resize[1]))
        CD20_array   = np.zeros((X_data.shape[0], tile_resize[0], tile_resize[1]))
        PaCK_array   = np.zeros((X_data.shape[0], tile_resize[0], tile_resize[1]))
        CD3_array    = np.zeros((X_data.shape[0], tile_resize[0], tile_resize[1]))
        CD68_array   = np.zeros((X_data.shape[0], tile_resize[0], tile_resize[1]))
        CD8_array    = np.zeros((X_data.shape[0], tile_resize[0], tile_resize[1]))
        for tile in range(X_data.shape[0]):
            nuclear_image = app.predict(np.expand_dims(nuclear[tile,:,:,:], 0), image_mpp=0.5, compartment='nuclear')
            PDL1_image    = app.predict(np.expand_dims(PDL1[tile,:,:,:],0), image_mpp=0.5, compartment='whole-cell')
            CD20_image    = app.predict(np.expand_dims(CD20[tile,:,:,:],0), image_mpp=0.5, compartment='whole-cell')
            PaCK_image    = app.predict(np.expand_dims(PaCK[tile,:,:,:],0), image_mpp=0.5, compartment='whole-cell')
            CD3_image     = app.predict(np.expand_dims(CD3 [tile,:,:,:],0), image_mpp=0.5, compartment='whole-cell')
            CD68_image    = app.predict(np.expand_dims(CD68[tile,:,:,:],0), image_mpp=0.5, compartment='whole-cell')
            CD8_image     = app.predict(np.expand_dims(CD8 [tile,:,:,:],0), image_mpp=0.5, compartment='whole-cell')
            
            X_data_array[tile,:,:,:] = X_data[tile,:,:,:]
            nuclear_array[tile,:,:]= nuclear_image[0,:,:,0]
            PDL1_array[tile,:,:]   = PDL1_image[0,:,:,0]
            CD20_array[tile,:,:]   = CD20_image[0,:,:,0]
            PaCK_array[tile,:,:]   = PaCK_image[0,:,:,0]
            CD3_array[tile,:,:]    = CD3_image[0,:,:,0]
            CD68_array[tile,:,:]   = CD68_image[0,:,:,0]
            CD8_array[tile,:,:]    = CD8_image[0,:,:,0]

            # rgb_images = create_rgb_image(np.expand_dims(X_data[tile,:,:,:2],0), channel_colors=['green', 'blue'])
            # overlay_nuclear = make_outline_overlay(rgb_data=rgb_images, predictions=np.expand_dims(nuclear_array[tile,:,:], (0,-1)))
            # overlay_cell = make_outline_overlay(rgb_data=rgb_images, predictions=np.expand_dims(PDL1_array[tile,:,:], (0,-1)))
            # cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/tmp/'+'{}_25_rbg.tif'.format(tile),rgb_images[0, :,:,:])
            # cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/tmp/'+'{}_25_nuc.tif'.format(tile),overlay_nuclear[0, :,:,:])
            # cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/tmp/'+'{}_25_cell.tif'.format(tile),overlay_cell[0, :,:,:])
            # sys.exit(0)
        
        new_dict ={}
        new_dict['X_data']      = X_data_array
        new_dict['nuclear_pred']= nuclear_array
        new_dict['PDL1_pred']   = PDL1_array
        new_dict['CD20_pred']   = CD20_array
        new_dict['PaCK_pred']   = PaCK_array
        new_dict['CD3_pred']    = CD3_array
        new_dict['CD68_pred']   = CD68_array
        new_dict['CD8_pred']    = CD8_array

        # print(new_dict['X_data'].shape, new_dict['nuclear_pred'].shape, new_dict['PDL1_pred'].shape)

        base = 'slide_' + str(slide_num) + '_core_' + str(rN) + '_' + str(cN)
        pickle_file = os.path.join(save_path_, base +'.p')
        with open(pickle_file, 'wb') as fp:
            pickle.dump(new_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # # create rgb overlay of image data for visualization
        # rgb_images = create_rgb_image(X_data[:,:,:,:2], channel_colors=['green', 'blue'])
        # overlay_nuclear = make_outline_overlay(rgb_data=rgb_images, predictions=app.predict(PDL1, image_mpp=0.5, compartment='nuclear'))
        # overlay_cell = make_outline_overlay(rgb_data=rgb_images, predictions=app.predict(PDL1, image_mpp=0.5, compartment='whole-cell'))
        # for i in range(X_data.shape[0]):
        #     cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/tmp/'+'{}{}_25_rbg.tif'.format(i, core_base_name),rgb_images[i, :,:,:])
        #     cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/tmp/'+'{}{}_25_nuc.tif'.format(i, core_base_name),overlay_nuclear[i, :,:,:])
        #     cv2.imwrite('/srv/scratch/z5315726/mIF/mesmer/tmp/'+'{}{}_25_cell.tif'.format(i, core_base_name),overlay_cell[i, :,:,:])
        # sys.exit(0)

