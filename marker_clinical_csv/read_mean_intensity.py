import numpy as np
import pandas as pd
import os
import sys
import tifffile
import cv2
import csv
from glob import glob
import pickle
from scipy import stats
from skimage.measure import regionprops
from math import pi

patient_csv = '/srv/scratch/z5315726/mIF/mesmer/data_file.xlsx'
patient_df = pd.read_excel(patient_csv)
patient_df['studyid'] = patient_df['studyid'].astype(str).str.strip().astype(int)
# print(list(patient_df['studyid']))

tile_size = (256, 256)
px = 1

def cell_idx_fn(image_in, min_cnr, left_px, right_px,):
    cell_area = image_in[min_cnr[0]:min_cnr[0] + abs(left_px[0]-right_px[0]),min_cnr[1]:min_cnr[1]+abs(left_px[1]-right_px[1])]
    if len(cell_area.flatten()) == 0 : return None
    vals,counts = np.unique(cell_area, return_counts=True)
    # print('vals, counts of unique masks in tile ', vals, counts)
    index_1 = vals[np.argmax(counts)]
    index_2 = np.partition(cell_area.flatten(), -2)[-2]

    if index_1 != 0: cell_idx = index_1
    elif index_2 != 0: cell_idx = index_2
    else: return None

    return cell_idx

def artifacts_fn(image_in, mask_px, size):
    #empty mask image 
    if np.sum(image_in) == 0: return image_in

    cell_mask_list = np.argwhere(image_in == mask_px)
    sum_ = cell_mask_list.sum(axis = 1)
    min_mask_px = cell_mask_list[np.argmin(sum_)]
    max_mask_px = cell_mask_list[np.argmax(sum_)]
    if ((abs(max_mask_px[1] - min_mask_px[1]) > 0.2*size[1]) or (abs(max_mask_px[0] - min_mask_px[0]) > 0.2*size[0])): 
        return np.where(image_in == mask_px, 0, 0)

    return image_in

def features_fn(input_mask, filter_cell, px):
    props_out = regionprops(filter_cell, input_mask)

    if props_out != []:
        cell_area = props_out[0].area
        cell_ctrd = props_out[0].centroid
        cell_intensity = props_out[0].intensity_mean
        cell_intensity_max = props_out[0].intensity_max
        cell_perimeter = props_out[0].perimeter
        cell_circularity = (4*pi*cell_area)/(cell_perimeter*cell_perimeter)
        cell_major_axis = props_out[0].axis_major_length
        cell_minor_axis = props_out[0].axis_minor_length

        # print('area centroid',cell_area, cell_ctrd)
        # print('intensity, intensity max, perimeter', cell_intensity, cell_intensity_max, cell_perimeter)
        # print('circularity', cell_circularity)
        # print('cell major minor axis', cell_major_axis, cell_minor_axis)

        return (cell_area, cell_ctrd, cell_intensity, cell_intensity_max, cell_perimeter, cell_circularity, cell_major_axis, cell_minor_axis)

    else:
        return (0,0,0,0, 0,0,0,0)


for slide_num in [9]:
    pickel_folder = '/srv/scratch/bic/piumi/mesmer/all_slides_new/slide{}/'.format(slide_num)
    TMA_csv = '/srv/scratch/z5315726/mIF/mesmer/TMA_{}.csv'.format(slide_num)
    TMA_df = pd.read_csv(TMA_csv, header=None)
    
    new_id = 1
    patient_list = {}

    MIAMI_clinical_csv = '/srv/scratch/z5315726/mIF/mesmer/marker_clinical_csv/Clinical_Data_{}.csv'.format(slide_num)
    clinical_f = open(MIAMI_clinical_csv, 'w')
    clinical_writer = csv.writer(clinical_f)
    clinical_header = ['Patient_num', 'ID', 'Recurrence', 'Recurrence_time', 'Survival', 'Survival_time', 'Age', 'Core', 'C', 'R']
    clinical_writer.writerow(clinical_header)
    
    for core in glob(pickel_folder+'*.p'):
        print(core)
        base = os.path.basename(core)
        _, slide_number, _, r_raw, c_char_raw = base.split('.p')[0].split('_')
        if int(slide_number) in [1,2,3,4,6,7,8,9]:
            r = r_raw
            c_char = c_char_raw
        elif int(slide_number) == 5:
            r = c_char_raw
            c_char = r_raw
        else: continue
        c = ord(c_char.lower()) - 96 - 1 # correction for 0th index 
        patient_number = TMA_df[c][int(r)-1] # correction for 0th index 
        if patient_number == 'Normal Breast' or patient_number == 'Spleen': continue
        if patient_number not in patient_list.keys(): 
            patient_list[patient_number] = new_id
            current_id = new_id
            new_id +=1
        else: current_id = patient_list[patient_number]
        try: data_file_index = patient_df[patient_df['studyid'] == int(patient_number)].index[0]
        except: 
            print(patient_number , ' removed from calculation.')
            continue
        #patient details completed. Write clinical csv
        surv = patient_df['bcdeath'][data_file_index]
        yearsfu = patient_df['yearsfu'][data_file_index]
        age = patient_df['ageatdiagnosis'][data_file_index]
        
        MIAMI_clinical_row = [patient_number, current_id+slide_num*1000, 0, 0, surv, yearsfu, age, base, c, int(r)-1]
        # print(MIAMI_clinical_row)
        clinical_writer.writerow(MIAMI_clinical_row)

        MIAMI_marker_csv = '/srv/scratch/z5315726/mIF/mesmer/marker_clinical_csv/marker_files/markers_{}_{}_{}_{}.csv'.format(slide_num, current_id+slide_num*1000, c, int(r)-1)
        marker_f = open(MIAMI_marker_csv, 'w')
        marker_writer = csv.writer(marker_f)
        marker_header = ['ID', 'patch_num', \
        'DAPI_area', 'DAPI_ctrd', 'DAPI_int', 'DAPI_int_max', 'DAPI_peri', 'DAPI_circ', 'DAPI_mj_ax', 'DAPI_mi_ax', \
        'PDL1_area', 'PDL1_ctrd', 'PDL1_int', 'PDL1_int_max', 'PDL1_peri', 'PDL1_circ', 'PDL1_mj_ax', 'PDL1_mi_ax', \
        'CD20_area', 'CD20_ctrd', 'CD20_int', 'CD20_int_max', 'CD20_peri', 'CD20_circ', 'CD20_mj_ax', 'CD20_mi_ax', \
        'PaCK_area', 'PaCK_ctrd', 'PaCK_int', 'PaCK_int_max', 'PaCK_peri', 'PaCK_circ', 'PaCK_mj_ax', 'PaCK_mi_ax', \
        'CD03_area', 'CD03_ctrd', 'CD03_int', 'CD03_int_max', 'CD03_peri', 'CD03_circ', 'CD03_mj_ax', 'CD03_mi_ax', \
        'CD68_area', 'CD68_ctrd', 'CD68_int', 'CD68_int_max', 'CD68_peri', 'CD68_circ', 'CD68_mj_ax', 'CD68_mi_ax', \
        'CD08_area', 'CD08_ctrd', 'CD08_int', 'CD08_int_max', 'CD08_peri', 'CD08_circ', 'CD08_mj_ax', 'CD08_mi_ax', \
        'left_px_0', 'left_px_1',\
        'right_px_0', 'right_px_1']
        marker_writer.writerow(marker_header)

        core_path = os.path.join(pickel_folder, core)
        with open(core_path, 'rb') as fp:
            data_file = pickle.load(fp)

        X_data = data_file['X_data']
        DAPI = data_file['nuclear_pred']
        PDL1 = data_file['PDL1_pred']
        CD20 = data_file['CD20_pred']
        PanCK = data_file['PaCK_pred']
        CD3 = data_file['CD3_pred']
        CD68 = data_file['CD68_pred']
        CD8 = data_file['CD8_pred']

        DAPI_image = X_data[:,:,:,0]
        PDL1_image = X_data[:,:,:,1]
        CD20_image = X_data[:,:,:,2]
        PanCK_image = X_data[:,:,:,3]
        CD3_image = X_data[:,:,:,4]
        CD68_image = X_data[:,:,:,5]
        CD8_image = X_data[:,:,:,6]

        for tile_num in range(DAPI.shape[0]):
            DAPI_cells = np.delete(np.unique(DAPI[tile_num,:,:]), 0)
            # print(DAPI_cells)

            for idx in DAPI_cells:
                zeros = np.zeros(DAPI[tile_num,:,:].shape)
                single_cell_DAPI = np.where(DAPI[tile_num,:,:] == idx, px, 0)
                cell_coord_list = np.argwhere(single_cell_DAPI == px)
                s = cell_coord_list.sum(axis = 1)
                left_pixel = cell_coord_list[np.argmin(s)]
                right_pixel = cell_coord_list[np.argmax(s)]
                if left_pixel[0] > right_pixel[0]: min_corner_0 = right_pixel[0]
                else : min_corner_0 = left_pixel[0]
                if left_pixel[1] > right_pixel[1]: min_corner_1 = right_pixel[1]
                else : min_corner_1 = left_pixel[1]

                cell_idx_pdl1 = cell_idx_fn(PDL1[tile_num,:,:], (min_corner_0, min_corner_1), left_pixel, right_pixel)
                cell_idx_cd20 = cell_idx_fn(CD20[tile_num,:,:], (min_corner_0, min_corner_1), left_pixel, right_pixel)
                cell_idx_pack = cell_idx_fn(PanCK[tile_num,:,:], (min_corner_0, min_corner_1), left_pixel, right_pixel)
                cell_idx_cd3  = cell_idx_fn(CD3[tile_num,:,:], (min_corner_0, min_corner_1), left_pixel, right_pixel)
                cell_idx_cd68 = cell_idx_fn(CD68[tile_num,:,:], (min_corner_0, min_corner_1), left_pixel, right_pixel)
                cell_idx_cd8  = cell_idx_fn(CD8[tile_num,:,:], (min_corner_0, min_corner_1), left_pixel, right_pixel)

                single_cell_PDL1 = np.where(PDL1[tile_num,:,:]  == cell_idx_pdl1, px, 0)
                single_cell_CD20 = np.where(CD20[tile_num,:,:]  == cell_idx_cd20, px, 0)
                single_cell_PaCK = np.where(PanCK[tile_num,:,:] == cell_idx_pack, px, 0)
                single_cell_CD03 = np.where(CD3[tile_num,:,:]   == cell_idx_cd3, px, 0)
                single_cell_CD68 = np.where(CD68[tile_num,:,:]  == cell_idx_cd68, px, 0)
                single_cell_CD08 = np.where(CD8[tile_num,:,:]   == cell_idx_cd8, px, 0)

                single_cell_PDL1 = artifacts_fn(single_cell_PDL1, px, tile_size)
                single_cell_CD20 = artifacts_fn(single_cell_CD20, px, tile_size)
                single_cell_PaCK = artifacts_fn(single_cell_PaCK, px, tile_size)
                single_cell_CD03 = artifacts_fn(single_cell_CD03, px, tile_size)
                single_cell_CD68 = artifacts_fn(single_cell_CD68, px, tile_size)
                single_cell_CD08 = artifacts_fn(single_cell_CD08, px, tile_size)

                # masked channels
                DAPI_cell_seg = DAPI_image[tile_num,:,:] *single_cell_DAPI
                PDL1_cell_seg = PDL1_image[tile_num,:,:] *single_cell_PDL1
                CD20_cell_seg = CD20_image[tile_num,:,:] *single_cell_CD20
                PaCK_cell_seg = PanCK_image[tile_num,:,:]*single_cell_PaCK
                CD03_cell_seg = CD3_image[tile_num,:,:]  *single_cell_CD03
                CD68_cell_seg = CD68_image[tile_num,:,:] *single_cell_CD68
                CD08_cell_seg = CD8_image[tile_num,:,:]  *single_cell_CD08

                # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/tmp/{}_{}_DAPI_.png".format(tile_num, idx), DAPI_image[tile_num,:,:]*single_cell_DAPI.astype(np.uint8))
                # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/tmp/{}_{}_PDL1_.png".format(tile_num, idx), PDL1_image[tile_num,:,:]*single_cell_PDL1.astype(np.uint8))
                # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/tmp/{}_{}_CD20_.png".format(tile_num, idx), CD20_image[tile_num,:,:]*single_cell_CD20.astype(np.uint8))
                # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/tmp/{}_{}_PaCK_.png".format(tile_num, idx), PanCK_image[tile_num,:,:]*single_cell_PaCK.astype(np.uint8))
                # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/tmp/{}_{}_CD3_.png".format(tile_num, idx),  CD3_image[tile_num,:,:]*single_cell_CD03.astype(np.uint8))
                # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/tmp/{}_{}_CD68_.png".format(tile_num, idx), CD68_image[tile_num,:,:]*single_cell_CD68.astype(np.uint8))
                # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/tmp/{}_{}_CD8_.png".format(tile_num, idx),  CD8_image[tile_num,:,:]*single_cell_CD08.astype(np.uint8))
                
                DAPI_feat = features_fn(DAPI_cell_seg, single_cell_DAPI, px)
                PDL1_feat = features_fn(PDL1_cell_seg, single_cell_PDL1, px)
                CD20_feat = features_fn(CD20_cell_seg, single_cell_CD20, px)
                PaCK_feat = features_fn(PaCK_cell_seg, single_cell_PaCK, px)
                CD03_feat = features_fn(CD03_cell_seg, single_cell_CD03, px)
                CD68_feat = features_fn(CD68_cell_seg, single_cell_CD68, px)
                CD08_feat = features_fn(CD08_cell_seg, single_cell_CD08, px)

                MIAMI_marker_row = [current_id+slide_num*1000, tile_num, \
                DAPI_feat[0], DAPI_feat[1], DAPI_feat[2], DAPI_feat[3], DAPI_feat[4], DAPI_feat[5], DAPI_feat[6], DAPI_feat[7], \
                PDL1_feat[0], PDL1_feat[1], PDL1_feat[2], PDL1_feat[3], PDL1_feat[4], PDL1_feat[5], PDL1_feat[6], PDL1_feat[7], \
                CD20_feat[0], CD20_feat[1], CD20_feat[2], CD20_feat[3], CD20_feat[4], CD20_feat[5], CD20_feat[6], CD20_feat[7], \
                PaCK_feat[0], PaCK_feat[1], PaCK_feat[2], PaCK_feat[3], PaCK_feat[4], PaCK_feat[5], PaCK_feat[6], PaCK_feat[7], \
                CD03_feat[0], CD03_feat[1], CD03_feat[2], CD03_feat[3], CD03_feat[4], CD03_feat[5], CD03_feat[6], CD03_feat[7], \
                CD68_feat[0], CD68_feat[1], CD68_feat[2], CD68_feat[3], CD68_feat[4], CD68_feat[5], CD68_feat[6], CD68_feat[7], \
                CD08_feat[0], CD08_feat[1], CD08_feat[2], CD08_feat[3], CD08_feat[4], CD08_feat[5], CD08_feat[6], CD08_feat[7], \
                left_pixel[0], left_pixel[1],\
                right_pixel[0], right_pixel[1]]

                # print(MIAMI_marker_row)
                marker_writer.writerow(MIAMI_marker_row)
            
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_DAPI.png".format(tile_num),DAPI_image[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_PDL1.png".format(tile_num),PDL1_image[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_CD20.png".format(tile_num),CD20_image[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_PaCK.png".format(tile_num),PanCK_image[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_CD3.png".format(tile_num), CD3_image[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_CD68.png".format(tile_num),CD68_image[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_CD8.png".format(tile_num), CD8_image[tile_num,:,:].astype(np.uint8))

            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_mask_DAPI.png".format(tile_num),DAPI[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_mask_PDL1.png".format(tile_num),PDL1[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_mask_CD20.png".format(tile_num),CD20[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_mask_PaCK.png".format(tile_num),PanCK[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_mask_CD3.png".format(tile_num), CD3[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_mask_CD68.png".format(tile_num),CD68[tile_num,:,:].astype(np.uint8))
            # cv2.imwrite("/srv/scratch/z5315726/mIF/mesmer/{}_mask_CD8.png".format(tile_num), CD8[tile_num,:,:].astype(np.uint8))
            
            # if tile_num == 2: sys.exit()

        marker_f.close()
    
    clinical_f.close()

