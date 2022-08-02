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
module_list = [os.path.abspath(os.path.join(x)) for x in ['..','../../Image_alignment']]
for module_path in module_list:
    if module_path not in sys.path:
        sys.path.append(module_path)
    print(module_path)
    
import align_helper
import importlib
importlib.reload(align_helper)

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
print(args_dict['barcode'])
print()





mag = '20x'
    
barcode = args_dict['barcode']
pickle_dir = 'pickles'
pickle_name = barcode+'_pickle.pickle'
pickle_path = pickle_dir + os.sep + pickle_name
print('\n\n'+pickle_path+'\n\n')
dfall = pd.read_pickle(pickle_path)



    
def os_swap(x):
    out = '/'+('/'.join(x.split('\\'))).replace('//','/')
    return out

dfall['parent_file'] = dfall['parent_file'].apply(lambda x: os_swap(x))



keeplist=[]

template_position_list = dfall.reset_index()['template_position'].unique()
# keylist = mag_dict.keys()
keylist = dfall.reset_index()['key'].unique()
# for Position in ['P2']:

print(template_position_list)
for Position in template_position_list: #go one position by position, since you need offsets per position
# for Position in ['P1', 'P3', 'P12', 'P15', 'P30', 'P31', 'P35']: #go one position by position, since you need offsets per position
    print('POSITION = ', Position)
    for key in keylist:
    


        dfr = dfall.reset_index().set_index(['template_position','key'])
        dfsub = dfr.loc[pd.IndexSlice[[Position],[key]],:]
        parent_file = dfsub['parent_file'][0]
        
        # print(key,Path(parent_file).name,Position)

        

        reader = AICSImage(parent_file)




        ##################
        #specify which channels to ke.ep
        ##################

        # channel_dict = get_channels(czi)
        channels = reader.channel_names
        # print('channels found = ', channels)

        
        align_channel = dfsub['align_channel'][0]
        # print('align_channel found = ', align_channel)
        
        position = Position
        scene = dfsub['Scene'][0]
        
        
        align_channel_index = [xi for xi,x in enumerate(channels) if x==align_channel][0]
        # print(position,' - S' + str(scene).zfill(3) )
        si = int(scene)-1 #scene_index

        reader.set_scene(si)
        try:
            T = reader.dims['T'][0]-1 #choose last time point
            notcorrupt = True
        except:
            notcorrupt = False
            print('CORRUPTED SCENE!!!!')
            
        if notcorrupt:
            align_chan = dfall.groupby('key').agg('first').loc[key,'align_channel']
            delayed_chunk = reader.get_image_dask_data("ZYX", T=T, C=align_channel_index)
            imgstack = delayed_chunk.compute()
            # print(imgstack.shape)
            keeplist.append((align_channel,key,imgstack))
        

    # now align images
    ### this workflow does not do camera alignment
    ### this workflow only does tranlsation (no scaling and no rotation)

    # for keep in keeplist:
    #     print(keep[1])

    
    subkeylist = [x[1] for x in keeplist]
    if len(keylist)>len(subkeylist):
        print('corrupted scene means only subset of rounds are exported')
        
    keylist = [x for x in keylist if x in subkeylist]
    
    
    img_list=[]
    for key in keylist:
        keep = [x for x in keeplist if x[1]==key][0]
        align_chan = keep[0]
        imgstack = keep[-1]
        # print(align_chan,key)
        img_list.append(imgstack.copy())
        # keeplist.append((channels,key,imgstack))
    len(img_list)

    # print([x.shape for x in img_list])
    offset_list = align_helper.find_xyz_offset_fromlist(img_list,ploton=False,verbose=False)
    # match_list = align_helper.return_aligned_img_list_new(img_list,offset_list)
    print("offset_list\n",offset_list)
    # unmatch_list = align_helper.return_aligned_img_list_new(img_list,np.asarray(offset_list)*0)

    
    roi_list = align_helper.return_aligned_img_list_new(img_list,offset_list,return_roi=True)


# now load up the full images (including the full timelapse)

        

    
    
    for ki,key in enumerate(keylist):
        print(key)

        dfr = dfall.reset_index().set_index(['template_position','key'])
        dfsub = dfr.loc[pd.IndexSlice[[Position],[key]],:]
        parent_file = dfsub['parent_file'][0]
        

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
                    # print(imgstack.shape)


                    roi = roi_list[ki]

                    the_crop_roi = np.index_exp[:, #don't crop in Z since max project to be done
                                        roi[2]:roi[3],
                                        roi[4]:roi[5]]

                    cropped_maxz = np.max(imgstack[the_crop_roi],axis=0) #compute max z projection

                    #now save this image




                    sep = os.sep
                    channel=c
                    channel_num = str(ci+1).zfill(2)
                    tnum = str(T).zfill(3)
                    savedir = f"{os.curdir}{sep}aligned_4i_exports{sep}{barcode}-export{sep}"

                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                        print('making',os.path.abspath(savedir))


                    savename = f"{barcode}-{mag}-R{round_num}-Scene-{scene_num}-P{position_num}-{well}-maxproj_c{channel_num}_T{tnum}_ORG.tif"
                    
                    # savename = f"{Path(file_path).stem}-S{str(scene+1).zfill(2)}-{channel}-T{str(T+1).zfill(4)}.tif"
                    # print(savedir,savename)
                    savepath = f"{savedir}{sep}{savename}"
                    # print(savepath)

                    skio.imsave(savepath,
                                np.uint16(cropped_maxz),
                                check_contrast=False)
                    if (T==0)&(ci==0):
                        print(os.path.abspath(savepath))

                