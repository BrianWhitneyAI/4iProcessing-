import math
import os
from typing import Tuple
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio import AICSImage, writers
import registration_utils
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import ast
import tifffile







def os_swap(x):
    out = '/'+('/'.join(x.split('\\'))).replace('//','/')
    return out




barcode = '5500000724'
yaml_dir = '/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/yml_configs'
pickle_files_dir = '/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/pickles'
output_example_dir = '/allen/aics/assay-dev/users/Goutham/4iProcessing-/4iprocessing/output_examples'


yaml_list = [x for x in os.listdir(yaml_dir) if x.endswith("confirmed.yaml")]
yaml_list = [x for x in yaml_list if barcode in x]
print("yaml list--- files that contain barcode are {}".format(yaml_list))
dflist =[]

for y in yaml_list:
    print(y)
    yml_path = yaml_dir + os.sep + y
    with open(yml_path) as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)
        for round_dict in data['Data']:
            #print(round_dict)
            # reader = AICSImage(iround_dict['path'])
            # channels = reader.channel_names
            # print(Path(iround_dict['path']).name)
            # print(data['Data'][0])
            # print()
            dfsub = pd.DataFrame(round_dict.values(), index=round_dict.keys()).T
            dfsub['barcode'] = data['barcode']
            dfsub['scope'] = data['scope']
            dfsub['output_path'] = data['output_path']
            dflist.append(dfsub)

print(f"df list is {dflist}")

dfconfig = pd.concat(dflist)
dfconfig.set_index(['barcode','round'],inplace=True)            
print("df config is:")
print(dfconfig)

dfconfig.to_csv("df_config_output.csv") # just list rounds,path.scenes, etc..


#output_dir = dfconfig['output_path'][0]
pickle_name = barcode+'_pickle.pickle'
pickle_path = pickle_files_dir + os.sep + pickle_name
print('\n\n'+pickle_files_dir+'\n\n')
dfall = pd.read_pickle(pickle_path)

#dfall.to_csv("dfall_config_output.csv") # defines positions and other

dfall['parent_file'] = dfall['parent_file'].apply(lambda x: os_swap(x))
print(dfall.columns)
dfall.to_csv("dfall_config_output_v2.csv") # defines positions and other
# issue with the first 2 columns not being detected?
dfall = pd.read_csv("/allen/aics/assay-dev/users/Goutham/4iProcessing-/4iprocessing/dfall_config_output_v2.csv")
print(dfall.columns)


position_chosen = 'P3'
target_img = 'Round 1'
source_img = 'Round 2'
roundlist = [target_img,source_img]
imglist=[]
### organizes images to be registered

for roundname in roundlist:
    dfline = dfall.loc[(dfall['template_position'] == position_chosen) & (dfall['key'] == roundname)]
    print("df line is {}".format(dfline))
    dfline.to_csv("dfline_example.csv")
    parent_file = Path(dfline['parent_file'].values[0])
    print("parent file is {}".format(parent_file))
    print(os.path.isfile(parent_file))
    reader = AICSImage(parent_file)

    scene = dfline['Scene']
    si = int(scene)-1 #scene_index
    reader.set_scene(si)

    #specify which channels to ke.ep
    ##################

    #channels = reader.dim
    channels_info= dfline['channel_dict'].values[0]
    dict_channels_info = ast.literal_eval(channels_info)
    channels = dict_channels_info['channel_names']
    align_channel = dfline['align_channel'].values[0]
    print(f"align channel is {align_channel}")
    channel_tuple_list = [(xi,x) for xi,x in enumerate(channels) if align_channel==x]
    T=0 # this needs to be iterated over?
    for ci,c in channel_tuple_list:
        delayed_chunk = reader.get_image_dask_data("ZYX",T=T,C=ci)
        imgstack = delayed_chunk.compute()
        imglist.append(imgstack)


target_img = imglist[0]
source_img = imglist[1]

print(np.shape(target_img))
print(np.shape(source_img))

print("padding")

target_img_padded = np.pad(target_img, ((0, 0), (50, 50), (50, 50)), mode='constant')

source_img_FOV = source_img[:, 150:np.shape(source_img)[1]-150, 150:np.shape(source_img)[1]-150]

source_img_padded = np.pad(source_img_FOV, ((0, 0), (50, 50), (50, 50)), mode='constant')

print(f"target img padded is of shape {np.shape(target_img_padded)}")
print(f"source img padded is of shape {np.shape(source_img_padded)}")

# # precropping
# target_img_FOV = target_img[:, 100:np.shape(target_img)[1]-100, 100:np.shape(target_img)[1]-100]
# source_img_FOV = source_img[:, 150:np.shape(source_img)[1]-150, 150:np.shape(source_img)[1]-150]

# target_img_z = target_img[10,:,:]
# source_img_z = source_img[10,:,:]

tifffile.imwrite(os.path.join(output_example_dir, "target_img_FOV.tiff"), target_img_padded)
tifffile.imwrite(os.path.join(output_example_dir, "source_img_FOV.tiff"), source_img_padded)

# Registration

source_aligned, target_aligned, composite = registration_utils.perform_alignment(source_img_padded, target_img_padded, smaller_fov_modality="source", scale_factor_xy=1, scale_factor_z=1, source_alignment_channel=0, target_alignment_channel=0, source_output_channel=[0], target_output_channel=[0], prealign_z=True, denoise_z=True)

print(f"source is {np.shape(source_aligned)}")
print(f"target img is of shape is {np.shape(target_aligned)}")
tifffile.imwrite("target_img_round_1_aligned.tiff", target_aligned)
tifffile.imwrite("source_img_round_2_aligned.tiff", source_aligned)



#tifffile.imwrite("composite.tiff", composite)
# composite = np.transpose(composite, axes=(3, 0, 1, 2))

# composite = np.transpose(composite, axes=(3, 0, 1, 2))
# with writers.OmeTiffWriter(
#     "composite.tiff",
#     overwrite_file=True,
# ) as writer:
#     writer.save(composite, dimension_order="CZYX")

# composite = np.transpose(composite, axes=(3, 0, 1, 2))

# writers.OmeTiffWriter.save(composite,  "composite.tiff", dim_order = 'CZYX')

# def perform_alignment(
#     source: np.ndarray,
#     target: np.ndarray,
#     smaller_fov_modality: str,
#     scale_factor_xy: float,
#     scale_factor_z: float,
#     source_alignment_channel: int,
#     target_alignment_channel: int,
#     source_output_channel: list,
#     target_output_channel: list,
#     prealign_z: bool,
#     denoise_z: bool,
#     use_refinement: bool,
#     save_composite: bool,
# ):