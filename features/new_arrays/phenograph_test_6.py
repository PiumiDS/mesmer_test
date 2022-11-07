import phenograph
import csv
import pandas as pd
import numpy as np
from glob import glob
import os, sys
import scanpy as sc
import scanpy.external as sce
import umap
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt
import networkx as nx

marker_csv_path = '/srv/scratch/z5315726/mIF/mesmer/features/new_arrays/markers/'
clinical_csv_path = '/srv/scratch/z5315726/mIF/mesmer/marker_clinical_csv/'
compress = umap.UMAP(n_neighbors=30)
tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=50)
clf = NearestCentroid()
k_val = 120

process_columns = [
		'DAPI_area', 'DAPI_peri', 'DAPI_c_x', 'DAPI_c_y', 'DAPI_circ', 'DAPI_mj_ax', 'DAPI_mn_ax', 'DAPI_int_mean', 'DAPI_int_max', 'DAPI_int_min', 'DAPI_int_std', \
		'PDL1_area', 'PDL1_peri', 'PDL1_c_x', 'PDL1_c_y', 'PDL1_circ', 'PDL1_mj_ax', 'PDL1_mn_ax', 'PDL1_int_mean', 'PDL1_int_max', 'PDL1_int_min', 'PDL1_int_std', 'PDL1_a_frac', \
		'CD20_area', 'CD20_peri', 'CD20_c_x', 'CD20_c_y', 'CD20_circ', 'CD20_mj_ax', 'CD20_mn_ax', 'CD20_int_mean', 'CD20_int_max', 'CD20_int_min', 'CD20_int_std', 'CD20_a_frac', \
		'PaCK_area', 'PaCK_peri', 'PaCK_c_x', 'PaCK_c_y', 'PaCK_circ', 'PaCK_mj_ax', 'PaCK_mn_ax', 'PaCK_int_mean', 'PaCK_int_max', 'PaCK_int_min', 'PaCK_int_std', 'PaCK_a_frac', \
		'CD03_area', 'CD03_peri', 'CD03_c_x', 'CD03_c_y', 'CD03_circ', 'CD03_mj_ax', 'CD03_mn_ax', 'CD03_int_mean', 'CD03_int_max', 'CD03_int_min', 'CD03_int_std', 'CD03_a_frac', \
		'CD68_area', 'CD68_peri', 'CD68_c_x', 'CD68_c_y', 'CD68_circ', 'CD68_mj_ax', 'CD68_mn_ax', 'CD68_int_mean', 'CD68_int_max', 'CD68_int_min', 'CD68_int_std', 'CD68_a_frac', \
		'CD08_area', 'CD08_peri', 'CD08_c_x', 'CD08_c_y', 'CD08_circ', 'CD08_mj_ax', 'CD08_mn_ax', 'CD08_int_mean', 'CD08_int_max', 'CD08_int_min', 'CD08_int_std', 'CD08_a_frac', \
		'cst_d', \
		'cst_DAPI_area', 'cst_DAPI_peri', 'cst_DAPI_circ', 'cst_DAPI_int_mean', \
		'cst_PDL1_area', 'cst_PDL1_peri', 'cst_PDL1_circ', 'cst_PDL1_int_mean', \
		'cst_CD20_area', 'cst_CD20_peri', 'cst_CD20_circ', 'cst_CD20_int_mean', \
		'cst_PaCK_area', 'cst_PaCK_peri', 'cst_PaCK_circ', 'cst_PaCK_int_mean', \
		'cst_CD03_area', 'cst_CD03_peri', 'cst_CD03_circ', 'cst_CD03_int_mean', \
		'cst_CD68_area', 'cst_CD68_peri', 'cst_CD68_circ', 'cst_CD68_int_mean', \
		'cst_CD08_area', 'cst_CD08_peri', 'cst_CD08_circ', 'cst_CD08_int_mean', \
		'ngb_DAPI_area', 'ngb_DAPI_peri', 'ngb_DAPI_circ', 'ngb_DAPI_int_mean', \
		'ngb_PDL1_area', 'ngb_PDL1_peri', 'ngb_PDL1_circ', 'ngb_PDL1_int_mean', \
		'ngb_CD20_area', 'ngb_CD20_peri', 'ngb_CD20_circ', 'ngb_CD20_int_mean', \
		'ngb_PaCK_area', 'ngb_PaCK_peri', 'ngb_PaCK_circ', 'ngb_PaCK_int_mean', \
		'ngb_CD03_area', 'ngb_CD03_peri', 'ngb_CD03_circ', 'ngb_CD03_int_mean', \
		'ngb_CD68_area', 'ngb_CD68_peri', 'ngb_CD68_circ', 'ngb_CD68_int_mean', \
		'ngb_CD08_area', 'ngb_CD08_peri', 'ngb_CD08_circ', 'ngb_CD08_int_mean', \
		]


def cell_closest_fn(array_xy, all_markers):
	d = cdist(array_xy, array_xy)
	sort_d = np.argsort(d)
	array_xy_closest = array_xy[sort_d[:, 1]]
	closest_array = all_markers[sort_d[:, 1]]
	closest_d = np.array(d[[i for i in range(len(array_xy))], list(sort_d[:,1])])
	coordinates = np.argwhere(d<50)
	coordinates = coordinates[coordinates[:,0].argsort()]
	values, indices = np.unique(coordinates[:,0], return_index=True)
	split_coord = np.delete(np.split(coordinates, indices), 0)
	mean_marker_values = np.array([np.mean(all_markers[split_coord[idx]][:,1], axis=0) for idx in range(split_coord.shape[0])])
	
	return closest_d, closest_array, mean_marker_values


for clinical_slide in glob(clinical_csv_path + 'Clinical_Data_*.csv'):
	slide_num = os.path.basename(clinical_slide).split('.csv')[0].split('_')[2]
	if int(slide_num) in [1,2,3,4,5,7,8,9]: continue
	clinical_df = pd.read_csv(clinical_slide)

	cluster_f_patient = open('/srv/scratch/z5315726/mIF/mesmer/features/new_arrays/cluster_info/cluster_info_{}_{}.csv'.format(int(slide_num), k_val), 'w')
	cluster_writer = csv.writer(cluster_f_patient)
	cluster_header = ['marker_path', 'n_neighbors', 'Q', 's_score']
	cluster_writer.writerow(cluster_header)

	p_ID_list = np.unique(clinical_df['ID'])
	for p_ID in p_ID_list:
		core_list_df = clinical_df[clinical_df['ID'] == p_ID]
		c_list = core_list_df['C'].values
		r_list = core_list_df['R'].values
		marker_files = ['markers_{}_{}_{}_{}.csv'.format(slide_num, p_ID, c_list[x], r_list[x]) for x in range(len(c_list))]

		for marker_file_path in [os.path.join(marker_csv_path, x) for x in marker_files]:
			marker_df = pd.read_csv(marker_file_path)
			marker_df['cell_num'] = np.arange(len(marker_df))
			x_coord = marker_df['DAPI_c_x']
			y_coord = marker_df['DAPI_c_y']
			coord_array = np.array([[x_coord[n], y_coord[n]] for n in range(len(x_coord))])
			arr_inp_closest = marker_df[[
					'DAPI_area', 'DAPI_peri', 'DAPI_circ', 'DAPI_int_mean', \
					'PDL1_area', 'PDL1_peri', 'PDL1_circ', 'PDL1_int_mean', \
					'CD20_area', 'CD20_peri', 'CD20_circ', 'CD20_int_mean', \
					'PaCK_area', 'PaCK_peri', 'PaCK_circ', 'PaCK_int_mean', \
					'CD03_area', 'CD03_peri', 'CD03_circ', 'CD03_int_mean', \
					'CD68_area', 'CD68_peri', 'CD68_circ', 'CD68_int_mean', \
					'CD08_area', 'CD08_peri', 'CD08_circ', 'CD08_int_mean', \
					]].to_numpy()
			closest_distance, closest_cell, neighbour_cells = cell_closest_fn(coord_array, arr_inp_closest)
			marker_df['cst_d'] = closest_distance
			marker_df[[
					'cst_DAPI_area', 'cst_DAPI_peri', 'cst_DAPI_circ', 'cst_DAPI_int_mean', \
					'cst_PDL1_area', 'cst_PDL1_peri', 'cst_PDL1_circ', 'cst_PDL1_int_mean', \
					'cst_CD20_area', 'cst_CD20_peri', 'cst_CD20_circ', 'cst_CD20_int_mean', \
					'cst_PaCK_area', 'cst_PaCK_peri', 'cst_PaCK_circ', 'cst_PaCK_int_mean', \
					'cst_CD03_area', 'cst_CD03_peri', 'cst_CD03_circ', 'cst_CD03_int_mean', \
					'cst_CD68_area', 'cst_CD68_peri', 'cst_CD68_circ', 'cst_CD68_int_mean', \
					'cst_CD08_area', 'cst_CD08_peri', 'cst_CD08_circ', 'cst_CD08_int_mean', \
					]] = closest_cell

			marker_df[[
					'ngb_DAPI_area', 'ngb_DAPI_peri', 'ngb_DAPI_circ', 'ngb_DAPI_int_mean', \
					'ngb_PDL1_area', 'ngb_PDL1_peri', 'ngb_PDL1_circ', 'ngb_PDL1_int_mean', \
					'ngb_CD20_area', 'ngb_CD20_peri', 'ngb_CD20_circ', 'ngb_CD20_int_mean', \
					'ngb_PaCK_area', 'ngb_PaCK_peri', 'ngb_PaCK_circ', 'ngb_PaCK_int_mean', \
					'ngb_CD03_area', 'ngb_CD03_peri', 'ngb_CD03_circ', 'ngb_CD03_int_mean', \
					'ngb_CD68_area', 'ngb_CD68_peri', 'ngb_CD68_circ', 'ngb_CD68_int_mean', \
					'ngb_CD08_area', 'ngb_CD08_peri', 'ngb_CD08_circ', 'ngb_CD08_int_mean', \
					]] = neighbour_cells

			marker_filtered = marker_df[process_columns]
			marker_array = marker_filtered.to_numpy()

			# ################################## PHENOGRAPH TEST
			# communities, graph, Q = phenograph.cluster(marker_array, k=60)
			# marker_vis = marker_filtered[[
			# 		'DAPI_area', 'DAPI_int', 'DAPI_int_max', 'DAPI_peri', 'DAPI_circ', 'DAPI_mj_ax', 'DAPI_mi_ax', \
			# 		'PDL1_area', 'PDL1_int', 'PDL1_int_max', 'PDL1_peri', 'PDL1_circ', 'PDL1_mj_ax', 'PDL1_mi_ax', \
			# 		'CD20_area', 'CD20_int', 'CD20_int_max', 'CD20_peri', 'CD20_circ', 'CD20_mj_ax', 'CD20_mi_ax', \
			# 		'PaCK_area', 'PaCK_int', 'PaCK_int_max', 'PaCK_peri', 'PaCK_circ', 'PaCK_mj_ax', 'PaCK_mi_ax', \
			# 		'CD03_area', 'CD03_int', 'CD03_int_max', 'CD03_peri', 'CD03_circ', 'CD03_mj_ax', 'CD03_mi_ax', \
			# 		'CD68_area', 'CD68_int', 'CD68_int_max', 'CD68_peri', 'CD68_circ', 'CD68_mj_ax', 'CD68_mi_ax', \
			# 		'CD08_area', 'CD08_int', 'CD08_int_max', 'CD08_peri', 'CD08_circ', 'CD08_mj_ax', 'CD08_mi_ax', \
			# 		]]
			# score = silhouette_score(marker_array, communities)
			# print('###############', k, Q, score, len(communities), len(np.unique(communities)))

			# # print(np.unique(communities, return_counts=True))
			# scaled_data = StandardScaler().fit_transform(marker_vis)
			# # print(scaled_data.shape)

			# for n in [4, 5, 7, 10, 20, 30, 40, 50]:
			# 	compress = umap.UMAP(n_neighbors=n, min_dist = 0.0)
			# 	embedding = compress.fit_transform(scaled_data)
			# 	# embedding = tsne.fit_transform(marker_vis, )
			# 	# print(embedding.shape, communities.shape)
			# 	plt.scatter(embedding[:,0], embedding[:,1], c=communities, cmap='Spectral', s=1)
			# 	plt.savefig('test_{}_{}.png'.format(os.path.basename(marker_file_path).split('.csv')[0], n))
			# 	plt.clf()

			# if p_ID == 7001: sys.exit()
			# ################################## PHENOGRAPH TEST

			################################ PHENOGRAPH VISUALISATION
			ret_data, graph, Q = phenograph.cluster(marker_array, k=k_val)
			marker_vis = marker_filtered[process_columns]
			score = silhouette_score(marker_array, ret_data)
			cluster_row = [marker_file_path, len(np.unique(ret_data)), Q, score]
			# print('unique clusters', len(np.unique(ret_data)), '\n', np.unique(ret_data))
			cluster_writer.writerow(cluster_row)

			pickle_file_c = '/srv/scratch/z5315726/mIF/mesmer/features/new_arrays/cluster_arrays/'+os.path.basename(marker_file_path).split('.csv')[0] +'.p'
			with open(pickle_file_c, 'wb') as fp_c:
				pickle.dump(np.array(ret_data), fp_c, protocol=pickle.HIGHEST_PROTOCOL)
			
			pickle_file_a = '/srv/scratch/z5315726/mIF/mesmer/features/new_arrays/cluster_arrays/'+os.path.basename(marker_file_path).split('.csv')[0] +'_array.p'
			with open(pickle_file_a, 'wb') as fp_a:
				pickle.dump(np.array(marker_filtered), fp_a, protocol=pickle.HIGHEST_PROTOCOL)

			pickle_file_b = '/srv/scratch/z5315726/mIF/mesmer/features/new_arrays/cluster_arrays/'+os.path.basename(marker_file_path).split('.csv')[0] +'_marker_df.p'
			with open(pickle_file_b, 'wb') as fp_b:
				pickle.dump(np.array(marker_df), fp_b, protocol=pickle.HIGHEST_PROTOCOL)
			
			# out = clf.fit(marker_array, ret_data)
			# centroids_list.extend(out.centroids_)
			# centroid_writer.writerow(out.centroids_)
			
			# # print(p_ID)
			# if p_ID > 1002: 
			# 	# cluster_f.close()
			# 	sys.exit()

			# scaled_data = StandardScaler().fit_transform(marker_vis)
			# compress = umap.UMAP(n_neighbors=k_val, min_dist = 0.005)
			# embedding = compress.fit_transform(scaled_data)
			# # embedding = tsne.fit_transform(scaled_data)
			# # print(embedding.shape, communities.shape)
			# plt.scatter(embedding[:,0], embedding[:,1], c=ret_data.tolist(), cmap='Spectral', s=1)
			# plt.xlabel('UMAP 1')
			# plt.ylabel('UMAP 2')
			# plt.savefig('/srv/scratch/z5315726/mIF/mesmer/features/new_arrays/png/cst_ngb_{}_{}_0.005.png'.format(os.path.basename(marker_file_path).split('.csv')[0], k_val))
			# plt.clf()
			# # if p_ID > 1000: sys.exit(0)
			# ################################ PHENOGRAPH VISUALISATION

# centroids_array = np.array(centroids_list)
# print(centroids_array.shape)

# centroid_communities, centroid_graph, centroid_Q = phenograph.cluster(centroids_array, k=20)

# scaled_data = StandardScaler().fit_transform(centroids_array)
# compress = umap.UMAP(n_neighbors=20, min_dist = 0.002)
# embedding = compress.fit_transform(scaled_data)
# # embedding = tsne.fit_transform(scaled_data)
# # print(embedding.shape, centroid_communities.shape)

# plt.scatter(embedding[:,0], embedding[:,1], c=centroid_communities.tolist(), cmap='Spectral', s=1)
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.savefig('/srv/scratch/z5315726/mIF/mesmer/features/slide2_20.png')
# plt.clf()
