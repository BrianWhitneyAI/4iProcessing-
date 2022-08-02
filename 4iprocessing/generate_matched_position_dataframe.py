from aicsimageio import AICSImage
from pathlib import Path
import  xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


import re
import os
import zen_position_helper
pd.set_option('display.max_columns',None)
ploton=False

# todo: parse argument to decide which barcode to run?


"""
0. find the yaml files for each barcode (currently finds all barcodes)

1. retrieve positions/scenes/coordinates from experiment file for acquired positions

2. remove scenes that are marked for removal

3. CHECK1: for each experiment --  determine which FOVs overlap and remove those overlapping FOVs. 
    a. in near future it would be good to keep the FOV that was acquired first. Actually this is a bad idea. 
    b. it is a bad idea because the overlap will cause bleaching in the images acquired with the next modality!!!!!
    c. At this point all the keepable FOVs are kept and all the incorrect/overlapping/problematic FOVs are removed
    d. FUTURE: dont remove these positions...instead they should be flagged so the user knows that there was an issue. 

4. Now set one imaging round to be the fixed "template"/"temp" dataframe/positionlist. 
    a. for each FOV in this template imaging round, look for FOVs in the subsequent imaging round that overlap to identify matched sets. 
    b. the same thing is said again in line 5.

5. The next step is to match positions of the subsequent imaging rounds to the template dataframe/positionlist.
    a. align all positions to one set of imaging that will be set as the template
    b. all other datasets will be aligned to this and indexed with the positon number of the template dataset
    c. TODO: check on how to handle this. right now if a round had an extra position that didn't match to the timelapse (or template round) then that would get tossed without any notification or comment. Data is just gone because its assumed to be superfluoes. 

6. Then after the matches are generated, assemble into a dataframe and:
    a. export the csv or pickle that defines the matched positions
    
    
The idea is that this dataframe contains all of the information needed for calling any other processing step. 
TODO: decide if we want any contact sheet-style outputs from this code that help the user know what is being done. 
IDEA: create max projection tiled output that looks to see if the same FOV was indeed imaged for all "putative" matched positions. 
"""

import yaml
from yaml.loader import SafeLoader
import pandas as pd
# Open the file and load the file
yaml_dir = 'yml_configs'
yaml_list = [x for x in os.listdir(yaml_dir) if '_confirmed' in x]
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
            dflist.append(dfsub)
            
dfconfig = pd.concat(dflist)
dfconfig.set_index(['barcode','iround'],inplace=True)            


import importlib
importlib.reload(zen_position_helper)

def create_rectangle(xyz,imgsize_um):
    """
    credit: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    """
    from collections import namedtuple
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    rect =  Rectangle(
                    xyz[0]-imgsize_um[1]/2,
                    xyz[1]-imgsize_um[0]/2,
                    xyz[0]+imgsize_um[1]/2,
                    xyz[1]+imgsize_um[0]/2,
                   )
    return rect

def intersection_area(a, b):  # returns None if rectangles don't intersect
    """"
    requires a ==> Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    credit: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    """
    area_a = (a.xmax-a.xmin)*(a.ymax-a.ymin)
    area_b = (b.xmax-b.xmin)*(b.ymax-b.ymin)
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy / np.min((area_a,area_b))
    else:
        return 0
    
def plot_position_rectangles(dfforplot,fs=12,figsize=(5,5)):
    from itertools import cycle

    
    dfkeep=dfforplot.copy()
    dfkeep.reset_index(inplace=True)

    #now plot the overlap of different files
    import matplotlib.pyplot as plt
    #make a scatter plot of positions
    npt = 1 #number of 20x positions to examine
    colorlist = ['k','r','c','y','g']
    coloriter = cycle(colorlist)
    color_dict = {x:next(coloriter) for xi,x in enumerate(dfkeep.parent_file.unique())}
    print(color_dict)
    for well,df0 in dfkeep.groupby('Well_id'):
        plt.figure(figsize=figsize)
        
        ncols = len(df0['parent_file'].unique())+1
        fig,ax = plt.subplots(nrows=1,ncols=ncols,figsize = (figsize[0]*ncols,figsize[1]),
                             sharex=True,sharey=True)
        axlist = ax.reshape(-1,)
        
        for i,(img_label,df1) in enumerate(df0.groupby('parent_file')):
    #             df1['X'] = df1['X_adjusted']
    #             df1['Y'] = df1['Y_adjusted']

    #             df1['X'] = df1['X_original']
    #             df1['Y'] = df1['Y_original']
            X = df1['X'].to_numpy()
            Y = df1['Y'].to_numpy()
            PositionList = df1['Position'].tolist()

            imgsize_um = df1['imgsize_um'].to_numpy()[0]
            w=imgsize_um[1]
            h=imgsize_um[0]

            for ii in [i, ncols-1]:
                plt.sca(axlist[ii])
                cycle=-1
                for xx,y,pos in zip(X,Y,PositionList):
                    plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor=color_dict[img_label],alpha=0.2,
                                    edgecolor=(0,0,0,0))
                    cycle=cycle+1
                    plt.text(xx,y,pos,
                             fontsize=fs)
                plt.title(well+'\n'+Path(img_label).stem,fontsize=fs)


                # plt.title(well,fontsize=fs)
                plt.axis('square')
#                 xlim = plt.xlim()
#                 ylim = plt.ylim()
                
#                 lims = [np.min([xlim[0],ylim[0]]),
#                        np.max([xlim[1],ylim[1]])]
#                 plt.xlim(lims)
#                 plt.ylim(lims)
        plt.show()
        
print('done')

########################################
#first aggregate all files into a dataframe with metadata
########################################
print('first aggregate all files into a dataframe with metadata')
# importlib.reload(zen_position_helper)


for barcode,dfcb in dfconfig.groupby(['barcode']):
    dfkeep = []
    for iround,dfcbr in dfcb.groupby(['iround']):
    # for key,value in mag_dict.items():
    #     display(value)

        print(barcode,iround)

        #determine if flist is filepaths or fms fileids
        file_list = []
        original_file_list = []
        list_of_files = dfcbr.path.tolist()
        for file in list_of_files:
            original_file_list.append(file)
            if os.path.exists(file):
                file_list.append(file)
            else: #try to find FMS id
                # file = fms.get_file_by_id(file)
                # file_list.append('/'+file.path)
                # print(file,'-->',file.path)
                print('not there...update your yaml')

        #get position info from all files in the file list
        dfl = []
        if len(file_list)>0:
            for file,filename  in zip(original_file_list,file_list):
                print(file,filename)
                dfsub = zen_position_helper.get_position_info_from_czi(filename)
                dfsub['align_channel'] = dfcbr['ref_channel'][0]
                dfsub['barcode']=barcode
                dfsub['key']=iround
                dfl.append(dfsub)


            df = pd.concat(dfl,
                           keys = original_file_list,
                           names = ['original_file'])





        dfkeep.append(df)



    # magkeylist = list(mag_dict.keys())
    # dfall = pd.concat(dfkeep,keys=magkeylist,names=['key']).reset_index()
    dfall = pd.concat(dfkeep).reset_index()
    dfall.set_index(['key'],inplace=True)
    
    #important columns are:
    # ['original_file',
    #  'file',
    #  'parent_file',
    #  'shape',
    #  'ImagePixelDistances',
    #  'totalmagnification',
    #  'channel_dict',
    #  'pixelSizes',
    #  'imgsize_um',
    #  'PlateAnchorPoint',
    #  'PlateReferencePoint',
    #  'X',
    #  'Y',
    #  'Z',
    #  'IsUsedForAcquisition',
    #  'Position',
    #  'Position_num',
    #  'Scene',
    #  'Well_id',
    #  'X_original',
    #  'X_adjusted',
    #  'Y_original',
    #  'Y_adjusted',
    #  'align_channel',
    #  'barcode',]

    
    # TODO: define why use anchor point in zen_position_helper
    # dfall[['X','X_original','X_adjusted','PlateReferencePoint','PlateAnchorPoint']]


    if ploton:
        plot_position_rectangles(dfall,fs=18,figsize=(10,10))
    

    # dfall.reset_index().set_index(['Position']).loc['P16']

# dfall[['original_file',
#   'file',
#  'parent_file',
#  'shape',
#  'ImagePixelDistances',
#  'totalmagnification',
#  'channel_dict',
#  'pixelSizes',
#  'imgsize_um',
#  'PlateAnchorPoint',
#  'PlateReferencePoint',
#  'X',
#  'Y',
#  'Z',
#  'IsUsedForAcquisition',
#  'Position',
#  'Position_num',
#  'Scene',
#  'Well_id',
#  'X_original',
#  'X_adjusted',
#  'Y_original',
#  'Y_adjusted',
#  'align_channel',
#  'barcode',]]

    if ploton:
        plot_position_rectangles(dfall,fs=18,figsize=(10,10))



    scenes_to_toss_list = [int(x)  for x in dfcbr['scenes_to_toss'].tolist() if x.isnumeric()]
    scenes_to_toss_list



    ###################################
    # now remove the scenes specified in the dictionary above (or config file int he future)
    ###################################
    print('TODO: remove every first scene from these 4i data')
    print('now remove the scenes specified in the dictionary above (or config file int he future)')

    original_file_AND_scenes_to_toss_list=[]
    # for key,value in mag_dict.items():
    for iround,dfcbr in dfcb.groupby(['iround']):
        # for key,value in mag_dict.items():
        #     display(value)

        # print(barcode,iround)


        # scenes_to_toss_list = value['scenes_to_toss']
        scenes_to_toss_list = [eval('['+x+']') for x in dfcbr['scenes_to_toss'].tolist()]
        original_file_list = dfcbr.path.tolist()
        for oi, original_file in enumerate(original_file_list):
            scenes_to_toss = scenes_to_toss_list[oi]
            print(Path(original_file).name, scenes_to_toss)
            original_file_AND_scenes_to_toss_list.extend([(original_file,scene) for scene in scenes_to_toss])


    # print(original_file_AND_scenes_to_toss_list)
    dfall.reset_index(inplace=True)
    dfall.set_index(['original_file','Scene'],inplace=True,)
    dfall.drop(labels= original_file_AND_scenes_to_toss_list,inplace=True,errors='ignore')

    # ploton=True
    if ploton:
        plot_position_rectangles(dfall)

    ###################################
    # find and record overlapping FOVs with magX and magX (self overlaps to be removed)
    ###################################
    positions_to_remove_list=[]
    print('find and record overlapping FOVs with magX and magX (self overlaps to be removed)')
    print('TODO: rename mag as key')
    for mag, df in dfall.groupby('key'):


        print(mag)
        dfl=[]
        for i,((pos,pf),dftemp_pos) in enumerate(df.groupby(['Position','parent_file'])):

            xyz = dftemp_pos[['X','Y','Z']].to_numpy()[0]
            imgsize_um = dftemp_pos['imgsize_um'].to_numpy()[0]
            template_rectangle =  create_rectangle(xyz,imgsize_um)

            for k,((pos2,pf2),dfmove_pos) in enumerate(df.groupby(['Position','parent_file'])):
                xyz2 = dfmove_pos[['X','Y','Z']].to_numpy()[0]
                imgsize_um2 = dfmove_pos['imgsize_um'].to_numpy()[0]
                move_rectangle =  create_rectangle(xyz2,imgsize_um2)


                # if (pos=='P16')&(mag=='20x_round6'):
                #     print(Path(pf).stem)

                overlap = intersection_area(template_rectangle,move_rectangle)
                # if (pos2=='P16')&(mag=='20x_round6')&(overlap>0):
                #     print(pos2,xyz,xyz2)
                #     print(pf,pf2)
                # (pos!=pos2) 
                
                #must overlap, and must not be the same position name (unless its from different parent files)
                
                # if the positions overlap AND (they don't have the same position name  OR they are from a different parent file) then mark them for removal. 
                if (overlap>0) & ((pos!=pos2) | (pf!=pf2)): 
                    feats={}
        #             print(pos,pos2,overlap)
                    feats['match_temp'] = pos
                    feats['match_move'] = pos2
                    feats['overlap'] = overlap
                    dfl.append(pd.DataFrame(data=feats.values(),index=feats.keys()).T)
        if len(dfl)>1: #needed to account for if no overlap occurs
            dfoverlap = pd.concat(dfl)
            positions_to_remove = list(set(dfoverlap.match_move.tolist() + dfoverlap.match_temp.tolist()))
            positions_to_remove_list.extend([(mag,position) for position in positions_to_remove])


    positions_to_remove_list    

    dfall.reset_index(inplace=True)
    dfall.set_index(['key', 'Position'], inplace=True)
    dfall.drop(labels = positions_to_remove_list,inplace=True, errors='ignore')

    # ploton=True
    if ploton:
        plot_position_rectangles(dfall)
        

    #define the list to start with timelapse
    ukeys = dfall.reset_index()['key'].unique()
    keylist  = [x for x in ukeys  if 'Time' in x] + [x for x in ukeys if 'Time' not in x]
    print(keylist)


    keeplist=[]
    dflall=[]

    for ki in range(0,len(keylist)):
        dfall.reset_index(inplace=True)
        dfall.set_index(['key'],inplace=True)
        pdslice =pd.IndexSlice[keylist[0]] #should be first round or time lapse...defined up above
        
        if bool(re.search('time',keylist[0],re.IGNORECASE)):
            print('template key = ', keylist[0])
        else:
            print(print('template key = ', keylist[0]))
            print('throw an error here', error_here_)
            
        dftemp = dfall.loc[pdslice,:] #template set to which other sets are matched to. 



        pdslice =pd.IndexSlice[keylist[ki]]
        dfmove = dfall.loc[pdslice,:] #set to be matched/"moved" to template


        print(ki,pdslice)
        ###################################
        # find and record overlapping FOVs
        ###################################

        print('find and record overlapping FOVs')

        dfl=[]
        for i,(pos,dftemp_pos) in enumerate(dftemp.groupby('Position')):
            xyz = dftemp_pos[['X','Y','Z']].to_numpy()[0]
            imgsize_um = dftemp_pos['imgsize_um'].to_numpy()[0]
            template_rectangle =  create_rectangle(xyz,imgsize_um)

            for k,(pos2,dfmove_pos) in enumerate(dfmove.groupby('Position')):
                xyz2 = dfmove_pos[['X','Y','Z']].to_numpy()[0]
                imgsize_um2 = dfmove_pos['imgsize_um'].to_numpy()[0]
                move_rectangle =  create_rectangle(xyz2,imgsize_um2)


                overlap = intersection_area(template_rectangle,move_rectangle)
                if overlap>0:
                    feats={}
        #             print(pos,pos2,overlap)
                    feats['match_temp'] = pos
                    feats['temp_name'] = keylist[0]
                    feats['match_move'] = pos2
                    feats['move_name'] = keylist[ki]
                    feats['temp_XYZ'] = xyz
                    feats['match_XYZ'] = xyz2
                    feats[f"mag{ki}_xy_offset_from_mag{'1'}_center"] = xyz2[0:-1]-xyz[0:-1] #move coordinates relative to template coordinates
                    feats['overlap'] = overlap
                    dfl.append(pd.DataFrame(data=feats.values(),index=feats.keys()).T)
        dfoverlap = pd.concat(dfl)
        dfoverlap
        
        

        ###################################
        #only keep positions that overlap
        ###################################

        for temp_move in ['temp','move']:
            keeplist.extend([tuple(x) for x in dfoverlap[[temp_move+'_name','match_'+temp_move]].to_numpy()])
        keepset = list(set(keeplist))
        dfall.reset_index(inplace=True)
        dfall.set_index(['key','Position'],inplace=True)
        dfkeep = dfall.loc[keepset]



        #merge the overlapping position info with the template dataframe
        #(remember that the first round of matching is the template with itself)
        dfsub = dfkeep.loc[pd.IndexSlice[[keylist[ki]],:]]
        dfm_move = pd.merge(dfsub.reset_index(),
                       dfoverlap[['match_temp','match_move']],
                       left_on = ['Position'],
                       right_on = ['match_move'],
                       suffixes=('','')
                      )

        dflall.append(dfm_move)


    dfout = pd.concat(dflall)
    dfout

    print('TODO: figure out smart way to keep positions that were imaged too many times\n plan is to keep the first image')

    dfcount = dfout.reset_index().groupby('match_temp').agg('count')
    number_of_rounds = len(dfout.key.unique())
    overlapping_poslist = np.unique(dfcount[dfcount['Position']>=number_of_rounds].index)
    print('keeping all these positions ',overlapping_poslist)

    # dfkeep = dfout.reset_index().set_index('match_temp').loc[overlapping_poslist] #modify to keep only positions that overalp at all rounds
    
    dfkeep = dfout.reset_index().set_index('match_temp') #keep positions even if they don't have an overlap each round
    dfkeep.reset_index(inplace=True)
    dfkeep['template_position'] = dfkeep['match_temp'] #rename this column to template position so it is clear what it is
    dfkeep.set_index(['key','template_position'],inplace=True)
    print(dfkeep.shape,dfout.shape)
    dfall = dfkeep.copy()
    dfkeep
    
    
# important columns
# index columns are important : 
# ['key', 'template_position'])
# ['match_temp', #position name in template file--this is identical to index column"template_position"
#  'match_move', #position name in "moving" file
#  'Position', #position name, this was used for merging
#  'Scene',
#  'file',
#  'parent_file',
#  'shape',
#  'ImagePixelDistances',
#  'totalmagnification',
#  'channel_dict',
#  'pixelSizes',
#  'imgsize_um',
#  'PlateAnchorPoint',
#  'PlateReferencePoint',
#  'X',
#  'Y',
#  'Z',
#  'IsUsedForAcquisition',
#  'Position_num',
#  'Well_id',
#  'X_original',
#  'X_adjusted',
#  'Y_original',
#  'Y_adjusted',
#  'align_channel',
#  'barcode',
#  ]

    ###################################
    # now split scenes and write out all the czi files as ome.tiffs
    ###################################

    pickle_dir = 'pickles'
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    pickle_name = barcode+'_pickle.pickle'
    pickle_path = pickle_dir + os.sep + pickle_name
    print('\n\n'+pickle_path+'\n\n')
    dfall.to_pickle(os.path.abspath(pickle_path))