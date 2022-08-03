import numpy as np
from numpy.linalg import inv
import re
import skimage.io as skio
from aicspylibczi import CziFile
import pandas as pd
import numpy as np
import aicsimageio.writers.ome_tiff_writer as ome_tiff_writer
overwrite=True

from aicsimageio import AICSImage


#####for alignment
import sys
import os
from pathlib import Path

######


# dfall.reset_index(inplace=True)
# dfall.set_index(['key','original_file','Position'],inplace=True)
# for (key,parent_file),dfsub in dfall.groupby(['key','parent_file']):

print(sys.argv)
arg_list = [(x.replace('--',''),i) for i,x in enumerate(list(sys.argv)) if bool(re.search('--',x))]
args_dict ={}

for keyval in arg_list:
    args_dict[keyval[0]] = sys.argv[keyval[1]+1]
print(args_dict)
print()

# args_dict['barcode'] = '5500000724'
print(args_dict['barcode'])
print()


    

import yaml
from yaml.loader import SafeLoader
import pandas as pd
# Open the file and load the file

barcode = args_dict['barcode']


yaml_dir = 'yml_configs'
yaml_list = [x for x in os.listdir(yaml_dir) if '_confirmed' in x]
yaml_list = [x for x in yaml_list if barcode in x]
dflist =[]
for y in yaml_list:
    print(y)
    yml_path = yaml_dir+os.sep+y
    with open(yml_path) as f:
        data = yaml.load(f, Loader=SafeLoader)
        for iround_dict in data['Data']:
            # print(iround_dict)
            # reader = AICSImage(iround_dict['path'])
            # channels = reader.channel_names
            # print(Path(iround_dict['path']).name)
            # print(data['Data'][0])
            # print()
            dfsub = pd.DataFrame(iround_dict.values(),index=iround_dict.keys()).T

            dfsub['barcode'] = data['barcode']
            dfsub['scope'] = data['scope']
            dfsub['output_path'] = data['output_path']
            dflist.append(dfsub)
            
dfconfig = pd.concat(dflist)
dfconfig.set_index(['barcode','iround'],inplace=True)            



mag = '20x'
    

output_dir = dfconfig['output_path'][0]
pickle_dir = output_dir + os.sep + 'pickles'
pickle_name = barcode+'_pickle.pickle'
pickle_path = pickle_dir + os.sep + pickle_name
print('\n\n'+pickle_path+'\n\n')
dfall = pd.read_pickle(pickle_path)




output_dir = dfconfig['output_path'][0]
align_pickle_dir = output_dir + os.sep + 'alignment_pickles'
align_pickle_name = barcode+'alignment_pickle.pickle'
align_pickle_path = align_pickle_dir + os.sep + align_pickle_name
dfalign = pd.read_pickle(align_pickle_path)
dfalign.reset_index(inplace=True)
dfalign.set_index(['key','Position'],inplace=True)



    
def os_swap(x):
    out = '/'+('/'.join(x.split('\\'))).replace('//','/')
    return out

dfall['parent_file'] = dfall['parent_file'].apply(lambda x: os_swap(x))


############################
##### define the functions
############################

from scipy.ndimage import affine_transform


# # https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf

# Translate:
# 
# 
# 1 0 0 dx
# 0 1 0 dy
# 0 0 1 dz
# 0 0 0 1
# 
# 




def get_align_matrix(alignment_offset):
    align_matrix = np.eye(4)
    for i in range(len(alignment_offset)):
        align_matrix[i,3] = alignment_offset[i]*-1
    align_matrix = np.int16(align_matrix)
    return align_matrix


def get_shift_to_center_matrix(img_shape,output_shape):
    # output_shape > img_shape should be true for all dimensions
    # and the difference divided by two needs to be a whole integer value

    shape_diff = np.asarray(output_shape)-np.asarray(img_shape)
    shift = shape_diff/2

    shift_matrix = np.eye(4)
    for i in range(len(shift)):
        shift_matrix[i,3] = shift[i]
    shift_matrix = np.int16(shift_matrix)
    return shift_matrix

########################
########################



keeplist=[]

template_position_list = dfall.reset_index()['template_position'].unique()
# keylist = mag_dict.keys()
keylist = dfall.reset_index()['key'].unique()
# for Position in ['P2']:

print(template_position_list)
# for Position in template_position_list: #go one position by position, since you need offsets per position
for Position in ['P1', 'P3', 'P12']: #go one position by position, since you need offsets per position
    print('POSITION = ', Position)

    testing_keylist = [x for x in keylist if 'Time' not in x]
    print(testing_keylist)
    imglist=[]
    for ki,key in enumerate(testing_keylist):
    # for ki,key in enumerate(keylist):
        print(key)

        dfr = dfall.reset_index().set_index(['template_position','key'])
        dfsub = dfr.loc[pd.IndexSlice[[Position],[key]],:]
        parent_file = dfsub['parent_file'][0]
        

        
        alignment_offset = dfalign.loc[pd.IndexSlice[key,Position],'alignment_offsets_xyz']
        final_shape = np.uint16([ 100,
                1248 + 1248/3,
                1848 + 1848/3,
              ])
            
            
        reader = AICSImage(parent_file)
        ##################
        #specify which channels to ke.ep
        ##################
        
        
        scene = dfsub['Scene'][0]
        ############3 get variables for file name
        round_num0 = re.search('time|Round [0-9]+',parent_file,re.IGNORECASE).group(0)
        
        
        round_num = round_num0.replace('Time','0').replace('Round ','').zfill(2)
        
        scene_num=str(scene).zfill(2)
        position_num =Position[1::].zfill(2)
        well = parent_file = dfsub['Well_id'][0]
        ############3 get variables for file name
        
        
        

        # channel_dict = get_channels(czi)
        channels = reader.channel_names

        position = Position
        si = int(scene)-1 #scene_index

        reader.set_scene(si)
        
        # Tn = reader.dims['T'][0] #choose last time point
        try:
            Tn = reader.dims['T'][0] #choose last time point
            notcorrupt = True
        except:
            notcorrupt = False
            print('CORRUPTED SCENE!!!! = ', Position)
            
        
        if notcorrupt:
            for T in range(Tn):
                
                for ci,c in enumerate(channels):
                    # print(T,ci)
                    # print("key=",key,"R=",round_num,"S=",scene_num,"T=",T+1,'C=',ci+1,c)

                    delayed_chunk = reader.get_image_dask_data("ZYX",T=T,C=ci)
                    imgstack = delayed_chunk.compute()
                    


                    # this is where the alignment is performed
                    print(alignment_offset)
                    align_matrix = get_align_matrix(alignment_offset)
                    shift_to_center_matrix = get_shift_to_center_matrix(imgstack.shape,final_shape)
                    combo = shift_to_center_matrix @ align_matrix

                    #aligned image
                    processed_volume = affine_transform(imgstack, 
                                                        np.linalg.inv(combo),
                                                        output_shape=final_shape,
                                                       order=0, # order = 0 means no interpolation...juust use nearest neighbor
                                                       )

                    # #unaligned image
                    # center_processed_volume = affine_transform(imgstack, 
                    #                                 np.linalg.inv(shift_to_center_matrix),
                    #                                 output_shape=final_shape,
                    #                                order=0, # order = 0 means no interpolation...juust use nearest neighbor
                    #                                )
                    
                    
                    
                    
                    # cropped_maxz = np.max(imgstack[the_crop_roi],axis=0) #compute max z projection

                    cropped_maxz = np.max(processed_volume,axis=0) #compute max z projection
                    #now save this image




                    
                    output_dir = dfconfig['output_path'][0]
                    
                    sep = os.sep
                    channel=c
                    channel_num = str(ci+1).zfill(2)
                    tnum = str(T).zfill(3)
                    savedir = f"{output_dir}{sep}max_projection_image_exports{sep}{barcode}-export{sep}"

                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                        print('making',os.path.abspath(savedir))


                    file_name_stem = Path(dfsub['parent_file'].iloc[0]).stem
                    savename = f"{file_name_stem}-{mag}-R{round_num}-Scene-{scene_num}-P{position_num}-{well}-maxproj_c{channel_num}_T{tnum}_ORG.tif"
                    
                    # savename = f"{Path(file_path).stem}-S{str(scene+1).zfill(2)}-{channel}-T{str(T+1).zfill(4)}.tif"
                    # print(savedir,savename)
                    savepath = f"{savedir}{sep}{savename}"
                    # print(savepath)

                    skio.imsave(savepath,
                                np.uint16(cropped_maxz),
                                check_contrast=False)
                    if (T==0)&(ci==0):
                        print(os.path.abspath(savepath))

                