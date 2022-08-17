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

from scipy.ndimage import affine_transform


def get_align_matrix(alignment_offset):
    align_matrix = np.eye(4)
    for i in range(len(alignment_offset)):
        align_matrix[i, 3] = alignment_offset[i] * -1
    align_matrix = np.int16(align_matrix)
    return align_matrix


def os_swap(x):
    out = '/'+('/'.join(x.split('\\'))).replace('//','/')
    return out

def pad_target_image(target_img, z_offset, y_offset, x_offset):
    padded_target_img = np.pad(target_img, ((z_offset,z_offset), (y_offset, y_offset), (x_offset, x_offset)) ,  mode='constant')
    return padded_target_img

def get_shift_to_center_matrix(img_shape, output_shape):
    # output_shape > img_shape should be true for all dimensions
    # and the difference divided by two needs to be a whole integer value

    shape_diff = np.asarray(output_shape) - np.asarray(img_shape)
    print(f"shape_diff is {shape_diff}")
    print(f"first component is {np.asarray(output_shape)}")
    print(f"second component is {np.asarray(img_shape)}")

    shift = shape_diff / 2
    print(f"shift is {shift}")

    shift_matrix = np.eye(4)
    for i in range(len(shift)):
        shift_matrix[i, 3] = shift[i]
    shift_matrix = np.int16(shift_matrix)
    return shift_matrix

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
pickle_name = barcode+'metadata_pickle.pickle'
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


position_chosen = 'P3' # used to be
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

#target_img_padded = np.pad(target_img, ((0, 0), (50, 50), (50, 50)), mode='constant')

source_img_cropped = source_img[:, 150:np.shape(source_img)[1]-150, 150:np.shape(source_img)[2]-150] # cropping b/c source img needs to be entirely contained in target in order for alignment to work

#tifffile.imwrite(os.path.join(output_example_dir, "target_img_FOV.tiff"), target_img)
#tifffile.imwrite(os.path.join(output_example_dir, "source_img_FOV.tiff"), source_img)

# Registration
source_aligned, target_aligned, composite, final_z_offset, final_y_offset, final_x_offset = registration_utils.perform_alignment(source_img_cropped, target_img, smaller_fov_modality="source",  source_alignment_channel=0, target_alignment_channel=0, source_output_channel=[0], target_output_channel=[0], scale_factor_xy=1, scale_factor_z=1, prealign_z=True, denoise_z=True)


# final_z_offset = 7
# final_y_offset = 202
# final_x_offset = 110

# TODO: save out the output version of these files here
# save original images for alignment

tifffile.imwrite("source_for_alignment.tiff", source_img_cropped)
tifffile.imwrite("target_for_alignment.tiff", target_img)





#print(f"source is {np.shape(source_aligned)}")
#print(f"target img is of shape is {np.shape(target_aligned)}")
# This part is just for visualization purposes
tifffile.imwrite("target_img_round_1_aligned.tiff", target_aligned)
tifffile.imwrite("source_img_round_2_aligned.tiff", source_aligned)

final_shape = np.uint16(
    np.asarray(
        [
            100,
            1248 + 1248 / 3,
            1848 + 1848 / 3,
        ]
    )
)


#target_img_padded = pad_target_image(target_img, z_pad_offset, y_pad_offset, x_pad_offset)


print(f"final z alignment is {final_z_offset}")
print(f"final y alignment is {final_y_offset}")
print(f"final x alignment is {final_x_offset}")

#alignment_offset = [final_z_offset-z_pad_offset, final_y_offset-y_pad_offset, final_x_offset-x_pad_offset]


alignment_offset = [final_z_offset, final_x_offset-150, final_y_offset-150] #


#alignment_offset = [-final_z_offset, final_y_offset-150 ,final_x_offset-150]
align_shift_list = [np.asarray([0, 0, 0]), alignment_offset] # THis is because first image is the refrence image


aligned_img_list = []
unaligned_img_list = []
count=0




# align_matrix = get_align_matrix(alignment_offset)

# shift_to_center_matrix = get_shift_to_center_matrix(imgstack.shape, final_shape)

# print(f"align matrix is {align_matrix}")
# print(f"shift_to_center_matrix  is {shift_to_center_matrix}")

# combo = shift_to_center_matrix @ align_matrix 
# print(f"combo is {combo}")

# processed_volume = affine_transform(imgstack, np.linalg.inv(combo), output_shape=final_shape, order=0)

# print(np.shape(processed_volume))

#tifffile.imwrite(os.path.join(output_example_dir, f"count_{1}.tiff"), processed_volume)


for alignment_offset, img in zip(align_shift_list, imglist):
    print(alignment_offset)
    align_matrix = get_align_matrix(alignment_offset)
    print(f"align matrix is {align_matrix}")
    shift_to_center_matrix = registration_utils.get_shift_to_center_matrix(np.shape(img), final_shape)
    combo = shift_to_center_matrix @ align_matrix
    print(f"shift to center matrix is {shift_to_center_matrix}")
    print(f"combo is {combo}")
    print(f"image stack is of shape {np.shape(img.shape)}")
    processed_volume = registration_utils.affine_transform(img, np.linalg.inv(combo), output_shape=final_shape, order=0)
    #print(f"img is {img}")
    #print(f"alignment offset is {alignment_offset}")
    tifffile.imwrite(os.path.join(output_example_dir, f"count_{count}.tiff"), processed_volume)
    count+=1

# align_matrix = get_align_matrix(alignment_offset)

# shift_matrix = get_shift_to_center_matrix(img_shape, final_shape)


# print(align_matrix)

# processed_volume = affine_transform(source_img, np.linalg.inv(align_matrix), output_shape=np.shape(target_img), order=0)

# tifffile.imwrite(os.path.join(output_example_dir, "processed_volume.tiff"), processed_volume)

# tifffile.imwrite(os.path.join(output_example_dir, "target_img_padded.tiff"), target_img)








