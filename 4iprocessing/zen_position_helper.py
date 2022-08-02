import os
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import numpy as np
from itertools import product
import lxml.etree as etree




def compute_adjusted_xy(df,overwrite=True):
    if 'X_original' not in df.columns.tolist():
        for xi,xyz in enumerate(['X','Y']):
            xyz0 =df[xyz].tolist()
            df[xyz+'_original'] = xyz0
            xyzadjust = []
            anchorlist = df['PlateAnchorPoint'].tolist()
            referencelist = df['PlateReferencePoint'].tolist()
            for i in range(df.shape[0]):
                xyzadjust.append(xyz0[i]-anchorlist[i][xi])
            df[xyz+'_adjusted'] = xyzadjust
            if overwrite:
                df[xyz] = xyzadjust
    else:
        print('already adjsuted')
    return df
            
        

def remove_overlapping_lowmag(dfmag1,adjust=False):
    #this block here will assigns each mag2 image position to a mag1 image position based on imaging region overlap determind from metadata
    #this block here will exclude positions for analysis if the regions of imaging overlapped (i.e. it identifies multiple mag1 positions for a given mag2 position)
    adjuststr = ''
    if adjust:
        adjuststr='_adjust'
        
    
    dfmag1['position_num'] = dfmag1['Position_num']
   
    imgsize_um_mag1 = np.asarray(dfmag1['imgsize_um'].iloc[0])
    plist = []
    elist = [] #mark for exclusion
    pdrops = []
    for i in range(dfmag1.shape[0]):
        print(i,end='\r')
        
        xyz = (dfmag1.iloc[i].at['X'], 
               dfmag1.iloc[i].at['Y'],
               dfmag1.iloc[i].at['Z'])

        contained_log  = np.bool_(np.zeros(dfmag1.shape[0]))
        for u in range(dfmag1.shape[0]):
            xyz2 = (dfmag1.iloc[u].at['X'], 
                    dfmag1.iloc[u].at['Y'],
                    dfmag1.iloc[u].at['Z'])
            
            loglist=[]
            for k in range(2):
                loglist.append((xyz[k]> (xyz2[k]-imgsize_um_mag1[k])) & (xyz[k]< (xyz2[k]+imgsize_um_mag1[k])))
            if i!=u:
                contained_log[u] = loglist[0] & loglist[1]
            
            
        pnum=np.where(contained_log)[0]
        if (len(pnum)!=0):
            elist.append(True)
            overlaps = np.asarray(list(pnum)+[i])
            # higher_overlaps = list(overlaps[np.argsort(overlaps)[1::]]) 
            # plist.extend(higher_overlaps)
            print('TODO: make this based on imaging time')
            #but for now you should just remove all overlaps
            higher_overlaps = list(overlaps[np.argsort(overlaps)[0::]]) 
            plist.extend(higher_overlaps)
        else:
            elist.append(False)
            

       
    keep_bool = np.bool_(np.ones(dfmag1.shape[0]))
    keep_bool[np.int32(np.unique(plist))] = False
    if np.sum(~keep_bool)>0:
        print('tossing because of self-overlap: \n',dfmag1[~keep_bool][['fname','Position']])
    else:
        print('no overlaps! :)')
    dfmag1 = dfmag1[keep_bool]
    print()
    print(dfmag1.shape)

    return dfmag1



def find_matching_positions_and_remove_duplicates_and_incompletes(dftemp,dfmove,drop_incompletes=True):
    import scipy.spatial.distance as scidist
    import matplotlib.pyplot as plt
    img_size_move = dfmove['imgsize_um'].iloc[0]
    img_size_temp = dftemp['imgsize_um'].iloc[0]

    # now you need to find all "dfmove"   cooridantes (that is the higher mag dataframe that could have more positons tiled onto a lower mag image) that are closer together than 1.2 the img_size of the dftemp
    keeplist = []
    move = dfmove[['X','Y','Z']].copy()
    xyz_all = move.to_numpy()
    xyz_group_list=[]
    for i in range(dfmove.shape[0]):
        xyz = np.float32(xyz_all[i,:])

        contained_log  = np.bool_(np.zeros(xyz_all.shape[0]))
        for u in range(xyz_all.shape[0]):
            xyz2 = np.float32(xyz_all[u,:])

            loglist=[]
            diff = np.abs(xyz-xyz2)
            for k in range(2):
                loglist.append((diff[k]<img_size_temp[k]*1.2/2))
            contained_log[u] = loglist[0] & loglist[1]

        pnum = np.where(contained_log)[0]
        xyz_mean = np.mean(xyz_all[pnum,:],axis=0)
        xyz_group_list.append(xyz_mean)

    xyz_group_array = np.asarray(xyz_group_list)

    for xyzi,xyz in enumerate(['X','Y','Z']):
        move[xyz+'_grouped'] = xyz_group_array[:,xyzi]

    for xyzi,xyz in enumerate(['X','Y','Z']):
        dfmove[xyz+'_grouped']  = move[xyz+'_grouped'] 


    temp = dftemp[['X','Y','Z']].copy()
    for xyz in ['X','Y','Z']:
        temp[xyz] =  temp[xyz] - temp.iloc[0].at[xyz] #normalize all positions to first position of template

    #now perform graph matching to find the coordinates reference point that makes all points overlap best
    lessthan_list=[]
    for refnum in range(dftemp.shape[0]):
        move = dfmove[['X_grouped','Y_grouped','Z_grouped']].copy()
        for xyz in ['X','Y','Z']:
            move[xyz] =  move[xyz+'_grouped'] - move.iloc[refnum].at[xyz+'_grouped']


        xyz_temp = np.float32(temp[['X','Y']].to_numpy())
        xyz_move = np.float32(move[['X','Y']].to_numpy())
        
        cmat = scidist.cdist(xyz_temp, xyz_move, metric='euclidean', )
        lessthan_idx = cmat<20
        lessthan_sum = np.sum(lessthan_idx)
        lessthan_values = cmat[lessthan_idx].reshape(-1,)
        lessthan_list.append((lessthan_sum,lessthan_values))

    lessthan_sum_array = np.asarray([x[0] for x in lessthan_list])
    lessthan_values_array = np.asarray([x[1] for x in lessthan_list])
    ltmin = np.max(lessthan_sum_array)
    ltmin_idx = np.asarray([i for i,x in enumerate(lessthan_sum_array) if x==ltmin])
    if np.sum(ltmin_idx)>1: #write code to choose the one with the least distance difference
        lvmin_idx = np.argmin([np.sum(x) for x in lessthan_values_array[ltmin_idx]])

    else:
        lvmin_idx=0
    refnum = ltmin_idx[lvmin_idx]
    print('refnum=',refnum)



    temp_img_size = dftemp.iloc[0].at['imgsize_um']
    move_img_size = dfmove.iloc[0].at['imgsize_um']

    #now redefine move coordinates based on determined "refnum"    
    move = dfmove[['X_grouped','Y_grouped','Z_grouped']].copy()
    for xyz in ['X','Y','Z']:
        move[xyz] =  move[xyz+'_grouped'] - move.iloc[refnum].at[xyz+'_grouped']

    for xyz in ['X','Y','Z']:
        dfmove[xyz+'_refmatched'] = move[xyz]
        
    plist = [] #position list
    elist = [] #mark for exclusion
    pdrops = [] #positions to drop because of overlap
    for i in range(move.shape[0]):
        print(i,end='\r')
        xyz = np.asarray((move.iloc[i].at['X'], 
               move.iloc[i].at['Y'],
               move.iloc[i].at['Z']))



        contained_log  = np.bool_(np.zeros(temp.shape[0]))
        for u in range(temp.shape[0]):
            xyz2 = np.asarray((temp.iloc[u].at['X'], 
                    temp.iloc[u].at['Y'],
                    temp.iloc[u].at['Z']))

            loglist=[]
            diff = np.abs(xyz-xyz2)
            for k in range(2):
                loglist.append((diff[k]<img_size_temp[k]*1.2/2))
            contained_log[u] = loglist[0] & loglist[1]


        pnum=np.where(contained_log)[0]
        if len(pnum)>1:
            print('overlaps with more than one position')
            elist.append(True)
            pdrops.extend(pnum)
    #             plist.append(pnum[0]+1)
            plist.append(dftemp.iloc[pnum[0]].at['position_num'])

        elif len(pnum)==0:
            print('no match found for scene =', dfmove.iloc[i].at['Scene'], ' , ', dfmove.iloc[i].at['Position'], ' , ', dfmove.iloc[i].at['fname'])
            plist.append('')
            elist.append(True)
        else:
            elist.append(False)
    #             plist.append(pnum[0]+1)
            plist.append(dftemp.iloc[pnum[0]].at['Position_num'])


    

    dfmove['elist'] = elist
    dfmove['mag1_position_number'] = plist
    dftemp['mag1_position_number'] = dftemp['position_num']

    print()
    print('dropping from dfmove (no match):', dfmove[dfmove['elist']==True]['Position'].unique())
    dfmove = dfmove[dfmove['elist']==False] #only keep non duplicates, and matches


    if drop_incompletes:
        dfcount = dfmove.groupby(['mag1_position_number']).count() #count to determine number of matching high mag tiles per low mag image
        drop_incomplete_list = dfcount.index.values[dfcount['file']<dfcount['file'].max()] #determine to drop everything that does not have complete tiled matches
        complete_bool = [True if x not in drop_incomplete_list else False for x in dfmove['mag1_position_number'].tolist()]
        print('dropping from dfmove', dfmove[~np.asarray(complete_bool)]['mag1_position_number'].unique())
        dfmove = dfmove[complete_bool]

    print('highmag kept =', dfmove['mag1_position_number'].unique())

    #now drop every mag1 position that is not in the mag2 dataframe
    mag1list = dfmove['mag1_position_number'].tolist()
    keeplist = [True if x in mag1list else False for x in dftemp['mag1_position_number'].tolist()]
    print('dropping from mag1', dftemp[~np.asarray(keeplist)]['mag1_position_number'].unique())
    dftemp = dftemp[keeplist]

    return dftemp,dfmove


def remove_duplicates_and_incompletes(dfmag1,dfmag2,drop_incompletes=True):
    #this block here will assigns each mag2 image position to a mag1 image position based on imaging region overlap determind from metadata
    #this block here will exclude positions for analysis if the regions of imaging overlapped (i.e. it identifies multiple mag1 positions for a given mag2 position)
    dfmag1['position_num'] = dfmag1['Position'].apply(lambda x: int(x[1::]))
    dfmag2['position_num'] = dfmag2['Position'].apply(lambda x: int(x[1::]))
    
    imgsize_um_mag2 = dfmag2['imgsize_um'].iloc[0]
    imgsize_um_mag1 = dfmag1['imgsize_um'].iloc[0]
    plist = []
    elist = [] #mark for exclusion
    pdrops = []
    for i in range(dfmag2.shape[0]):
        print(i,end='\r')
        xyz = (dfmag2.iloc[i].at['X'], 
               dfmag2.iloc[i].at['Y'],
               dfmag2.iloc[i].at['Z'])

        contained_log  = np.bool_(np.zeros(dfmag1.shape[0]))
        for u in range(dfmag1.shape[0]):
            xyz2 = (dfmag1.iloc[u].at['X'], 
                    dfmag1.iloc[u].at['Y'],
                    dfmag1.iloc[u].at['Z'])
            
            loglist=[]
            for k in range(2):
                loglist.append((xyz[k]> (xyz2[k]-imgsize_um_mag1[k]/2)) & (xyz[k]< (xyz2[k]+imgsize_um_mag1[k]/2)))
            contained_log[u] = loglist[0] & loglist[1]
            
        
        pnum=np.where(contained_log)[0]
        if len(pnum)>1:
            print('overlaps with more than one position')
            elist.append(True)
            pdrops.extend(pnum)
#             plist.append(pnum[0]+1)
            plist.append(dfmag1.iloc[pnum[0]].at['position_num'])
            
        elif len(pnum)==0:
            print('no match found for scene =', dfmag2.iloc[i].at['Scene'], ' , ', dfmag2.iloc[i].at['Position'], ' , ', dfmag2.iloc[i].at['fname'])
            plist.append('')
            elist.append(True)
        else:
            elist.append(False)
#             plist.append(pnum[0]+1)
            plist.append(dfmag1.iloc[pnum[0]].at['position_num'])
            

       

    dfmag2['elist'] = elist
    dfmag2['mag1_position_number'] = plist
    dfmag1['mag1_position_number'] = dfmag1['position_num']
    
    print()
    print('dropping from mag2 (no match):', dfmag2[dfmag2['elist']==True]['Position'].unique())
    dfmag2 = dfmag2[dfmag2['elist']==False] #only keep non duplicates, and matches
    
    
    if drop_incompletes:
        dfcount = dfmag2.groupby(['mag1_position_number']).count() #count to determine number of matching high mag tiles per low mag image
        drop_incomplete_list = dfcount.index.values[dfcount['file']<dfcount['file'].max()] #determine to drop everything that does not have complete tiled matches
        complete_bool = [True if x not in drop_incomplete_list else False for x in dfmag2['mag1_position_number'].tolist()]
        print('dropping from mag2', dfmag2[~np.asarray(complete_bool)]['mag1_position_number'].unique())
        dfmag2 = dfmag2[complete_bool]
    
    print('highmag kept =', dfmag2['mag1_position_number'].unique())

    #now drop every mag1 position that is not in the mag2 dataframe
    mag1list = dfmag2['mag1_position_number'].tolist()
    keeplist = [True if x in mag1list else False for x in dfmag1['mag1_position_number'].tolist()]
    print('dropping from mag1', dfmag1[~np.asarray(keeplist)]['mag1_position_number'].unique())
    dfmag1 = dfmag1[keeplist]

    return dfmag1,dfmag2


# def semi_auto_based_on_tilenumbers(dfmag1,dfmag2,n=9):
#     #specific for 100x_lamin_h2b_0807
    
#     #this block here will assigns each mag2 image position to a mag1 image position based on imaging region overlap determind from metadata
#     #this block here will exclude positions for analysis if the regions of imaging overlapped (i.e. it identifies multiple mag1 positions for a given mag2 position)
#     dfmag1['position_num'] = dfmag1['Position'].apply(lambda x: int(x[1::]))
#     dfmag2['position_num'] = dfmag2['Position'].apply(lambda x: int(x[1::]))
    
#     mag1_pos=[]
#     for i in range(dfmag2.shape[0]):
#         pnum2 = dfmag2.iloc[i].at['position_num']
#         for u in range(dfmag1.shape[0]):
#             pnum1 = dfmag1.iloc[u].at['position_num']
#             pnum2_pred = (pnum1-1)*n + 1
#             if (pnum2> pnum2_pred ) & (pnum2< pnum2_pred+n-1):
#                 mag1_pos.append(pnum1)

#     dfmag2['mag1_position_number'] = mag1_pos
#     dfmag1['mag1_position_number'] = dfmag1['position_num']
    
#     print('dropping', dfmag2[dfmag2['elist']==True]['mag1_position_number'].unique())

    
#     dfcount = dfmag2.groupby('mag1_position_number').count() #count to determine number of matching high mag tiles per low mag image
#     droplist = dfcount.index.values[dfcount['file']<dfcount['file'].max()] #determine to drop everything that does not have complete tiled matches
#     keeplist = [True if x not in droplist else False for x in dfmag2['mag1_position_number'].tolist()]
#     print('dropping from mag2', dfmag2[~np.asarray(keeplist)]['mag1_position_number'].unique())
#     dfmag2 = dfmag2[keeplist]
#     print('highmag kept =', dfmag2['mag1_position_number'].unique())

#     #now drop every mag1 position that is not in the mag2 dataframe
#     mag1list = dfmag2['mag1_position_number'].tolist()
#     keeplist = [True if x in mag1list else False for x in dfmag1['mag1_position_number'].tolist()]
#     print('dropping from mag1', dfmag1[~np.asarray(keeplist)]['mag1_position_number'].unique())
#     dfmag1 = dfmag1[keeplist]

#     return dfmag1,dfmag2

def get_adjusted_positions_from_metadata(filename):
    from aicsimageio import AICSImage
    reader = AICSImage(filename)
    meta0 = reader.metadata    
    #convert metadata to lxml
    metastr = ET.tostring(meta0).decode("utf-8")
    meta = etree.fromstring(metastr) 
    print(meta.find('.//Template/AnchorPoint').text)
    print(meta.find('.//Template/ReferencePoint').text)
    print(meta.find('.//Template/ShapeDistanceX').text)
    print(meta.find('.//Template/ShapeDistanceY').text)
#     ET.dump(meta)
# <Template Name="Multiwell 96">
#          <AnchorPoint>-46227.399,-43291.15</AnchorPoint>
#          <BoundsSize>127800,85500</BoundsSize>
#          <Category>Multiwell</Category>
#          <HasRegularShapes>true</HasRegularShapes>
#          <ReferencePoint>63900,42750</ReferencePoint>
#          <ReferencePointLocationMode>Center</ReferencePointLocationMode>
#          <ShapeType>Ellipse</ShapeType>
#          <ShapeColumns>12</ShapeColumns>
#          <ShapeRows>8</ShapeRows>
#          <ShapeWidth>6211.187</ShapeWidth>
#          <ShapeHeight>6211.187</ShapeHeight>
#          <ShapeDistanceX>8996.945</ShapeDistanceX>
#          <ShapeDistanceY>8981.045</ShapeDistanceY>
#          <OriginalShapeWidth>5600</OriginalShapeWidth>
#          <OriginalShapeHeight>5600</OriginalShapeHeight>
#          <OriginalShapeDistanceX>9000</OriginalShapeDistanceX>
#          <OriginalShapeDistanceY>9000</OriginalShapeDistanceY>
#          <AreShapesCenteredX>true</AreShapesCenteredX>
#          <AreShapesCenteredY>true</AreShapesCenteredY>
#          <ShapesDistanceToBorderX>0</ShapesDistanceToBorderX>
#          <ShapesDistanceToBorderY>0</ShapesDistanceToBorderY>
#          <Rotation>0.116</Rotation>
#          <ImmersionPoint>13000,13000</ImmersionPoint>
#          <BottomMaterial>Glass</BottomMaterial>
#          <BottomThickness>170</BottomThickness>
#          <BottomOffset>1500</BottomOffset>
#          <BottomOffsetMeasured>NaN</BottomOffsetMeasured>
#          <BottomRefractiveIndex>1.52</BottomRefractiveIndex>
#          <InsertType>Default</InsertType>
#          <PartOfInsertName />
#          <BottomColor>Unknown</BottomColor>
#          <UsageHints />
#          <SupportPoints />
#         </Template>
    

def toss_scenes(df,scenes_to_toss_dict):
    dflist=[]
    toss_bool = np.bool_(np.zeros(df.shape[0]))
    for fname,scenelist in scenes_to_toss_dict.items():
        
        
        log = (np.asarray(df['fname']==fname)) & (np.asarray(df['Scene'].apply(lambda x: int(x) in scenelist)))
        toss_bool = np.logical_or(toss_bool,log)
        
    
    dfout = df[~toss_bool]
    print('tossing scenes = ', df[toss_bool]['Scene'].tolist())
    print('tossing positions = ', df[toss_bool]['Position'].tolist())

    return dfout

def toss_scenes_old(df,scenes_to_toss_dict):
    dflist=[]
    for fname,scenelist in scenes_to_toss_dict.items():
        
        df['fname']=df['file'].apply(lambda x: Path(x).stem)
        dfsub = df[df['fname']==fname]
        dfsub['toss_bool'] = dfsub['Scene'].apply(lambda x: int(x) in scenelist)
        dflist.append(dfsub)
    dfkeeps = pd.concat(dflist)
    dfout = dfkeeps[dfkeeps['toss_bool']==False]
    print('tossing scenes = ', dfkeeps[dfkeeps['toss_bool'] == True]['Scene'].tolist())
    print('tossing positions = ', dfkeeps[dfkeeps['toss_bool'] == True]['Position'].tolist())

    return dfout
    
def get_position_info_from_czilist(filelist):
    dflist=[]
    for filename  in filelist:
        try:
            dfsub = get_position_info_from_czi(filename)
        except:
            dfsub = get_position_info_from_czi_black(filename)
        dflist.append(dfsub)
    df = pd.concat(dflist)
    print(df.shape)
    return df

def get_position_info_from_czi(filename):
    from aicsimageio import AICSImage
    reader = AICSImage(filename)
    meta0 = reader.metadata    
    #convert metadata to lxml
    metastr = ET.tostring(meta0).decode("utf-8")
    meta = etree.fromstring(metastr) 
       

    info_attrib_list = ['Name','X','Y','Z','Id','IsUsedForAcquisition','imgsize_um','shape','pixelSizes','CameraPixelAccuracy','parent_file']
    
    feats={}
    feats['file'] = filename
    feats['parent_file'] = filename
    #find the image dimensions of the image
#     number_of_scenes_acquired0 = [y for x,y in zip(reader.dims,reader.size()) if x=='S'][0]
    number_of_scenes_acquired = eval(meta.find('.//SizeS').text)
#     if number_of_scenes_acquired0!=number_of_scenes_acquired:
#         print('metadata error! cannot retrieve all scenes acquired!')

    SizeZ = eval(meta.find('.//SizeZ').text)
    #get camera dimensions
    regions = list(meta.findall('.//ParameterCollection/ImageFrame'))
    txtout = regions[0].text
    frame_size_pixels = eval(txtout)
    
    #number of pixels in each dimension
    feats['shape'] = tuple((frame_size_pixels[-2],frame_size_pixels[-1],SizeZ)) #number of pixels in each dimension

    #find key imaging parameters
    # ImagePixelDistancesList = meta.findall('.//ImagePixelDistances')
    # ImagePixelDistancesList = meta.findall('.//ParameterCollection/ImagePixelDistances')
    ImagePixelDistancesList = meta.findall('.//ParameterCollection/ImagePixelDistances')
    # print('runnig')
    # totalmagnification = eval(totalmagnification.text)
    for ip in ImagePixelDistancesList[0:1]: #only choose the first camera
        feats['ImagePixelDistances'] = tuple(eval(ip.text))
        # ET.dump(ip)
        # print('dumping')
        feats['totalmagnification'] = eval(ip.getparent().find('./TotalMagnification').text)
        ImageFrame =  eval(ip.getparent().find('./ImageFrame').text)
        feats['CameraPixelAccuracy'] =  eval(ip.getparent().find('./CameraPixelAccuracy').text)
        
    channels = meta.findall('.//Information/Image/Dimensions/Channels/Channel')
    keeplist=[]
    channel_dict = {'channel_indices':[],
                   'channel_names':[]}    
    for channel in channels:
        index = channel.attrib['Id'].replace('Channel:','')
        channel_name = channel.attrib['Name']
        channel_dict['channel_indices'].append(index)
        channel_dict['channel_names'].append(channel_name)
        exposure_time = int(channel.find('.//ExposureTime').text)/1e6
        binning = channel.find('.//Binning').text
            
        try: 
            laser_wavelength = channel.find('.//ExcitationWavelength').text
            laser_intensity = channel.find('.//Intensity').text
        except:
            laser_wavelength = 'None'
            laser_intensity = 'None'

        feats['laser_wavelength'+'_'+channel_name] = laser_wavelength
        feats['exposure_time'+'_'+channel_name] = exposure_time
        feats['binning'] = binning
        feats['laser_intensity'+'_'+channel_name] = laser_intensity
    feats['channel_dict'] = str(channel_dict)
        
    ZStep = meta.find('.//Z/Positions/Interval/Increment')
    ZStep = eval(ZStep.text)
    xypxsize = (np.asarray(feats['ImagePixelDistances'])/feats['totalmagnification'])
    feats['pixelSizes'] = (xypxsize[0],xypxsize[1],ZStep)#units of um
    feats['imgsize_um'] = tuple([x*y for x,y in zip(feats['pixelSizes'] ,feats['shape'])])
    feats['PlateAnchorPoint'] =  eval('[' + meta.find('.//Template/AnchorPoint').text + ']')
    feats['PlateReferencePoint'] = eval('[' + meta.find('.//Template/ReferencePoint').text + ']')
    
    dfmetalist=[]
    cycle=0
    for regions in [meta.find('.//SingleTileRegions')]: 
        #some weird czis have duplicates in SingleTileRegions....so you need to drop those by not doing find all
        for region in regions.findall('SingleTileRegion'):

            attrib = region.attrib
            feats['Name'] = attrib['Name']
            Id = str(attrib['Id'])
            for info in region.findall('X'):
                feats['X'] = float(info.text)
            for info in region.findall('Y'):
                feats['Y'] = float(info.text)
            for info in region.findall('Z'):
                feats['Z'] = float(info.text)
            for info in region.findall('IsUsedForAcquisition'):
                feats['IsUsedForAcquisition'] = info.text



            dfmetalist.append(pd.DataFrame(data=feats.values(),index=feats.keys()).T)
    df1 = pd.concat(dfmetalist)

    #now search for next set of parameters from .//Scene xml region
    #this pulls out the differences between how scene number and position number!
    dfsp=[]
    info_val_list=[]
    info_attrib_list = ['Position','Scene','Well_id']
    for region in meta.findall('.//Scene'):
        feats={}
        attrib = region.attrib
        feats['Position'] = attrib['Name']
        feats['Position_num'] = int(feats['Position'].replace('P',''))
        feats['Scene'] = int(attrib['Index'])+1
        subregion = region.find('Shape')
        feats['Well_id'] = 'unknown' #in case Shape is not a feature
        for subregion in region.findall('Shape'):
            feats['Well_id'] = subregion.attrib['Name']
        dfsp.append(pd.DataFrame(data = feats.values(), index = feats.keys()).T)
        
    df2 = pd.concat(dfsp)
    df = pd.merge(df1,df2,left_on='Name',right_on='Position',suffixes=('_1','_2'))
    dfsub =df[df['IsUsedForAcquisition']=='true']
    dfsub = dfsub[dfsub['Scene'].astype(int)<=number_of_scenes_acquired]
    dfsub['fname']=dfsub['file'].apply(lambda x: Path(x).stem)
    dfsub = compute_adjusted_xy(dfsub)
    
    
    
    return dfsub


def get_position_info_from_czi_black(filename):
    from aicsimageio import AICSImage
    reader = AICSImage(filename)
    meta0 = reader.metadata    
    #convert metadata to lxml
    metastr = ET.tostring(meta0).decode("utf-8")
    meta = etree.fromstring(metastr) 
    print()
    print('reading from zen black')
    print()


    info_attrib_list = ['Name','X','Y','Z','Id','IsUsedForAcquisition','imgsize_um','shape','pixelSizes','CameraPixelAccuracy','parent_file']

    feats={}
    feats['file'] = filename
    feats['parent_file'] = filename

    #find the image dimensions of the image
    number_of_scenes_acquired = eval(meta.find('.//SizeS').text)

    SizeZ = eval(meta.find('.//SizeZ').text)
    SizeX = eval(meta.find('.//SizeX').text)
    SizeY = eval(meta.find('.//SizeY').text)
    feats['shape'] = tuple((SizeX,SizeY,SizeZ)) #number of pixels in each dimension


    #find key imaging parameters
    ImagePixelDistancesList = meta.findall('.//ImagePixelDistances')
    for ip in ImagePixelDistancesList[0:1]: #only choose the first camera
        feats['ImagePixelDistances'] = tuple(eval(ip.text))
        feats['totalmagnification'] = eval(ip.getparent().find('./TotalMagnification').text)
        ImageFrame =  eval(ip.getparent().find('./ImageFrame').text)
        feats['CameraPixelAccuracy'] =  eval(ip.getparent().find('./CameraPixelAccuracy').text)

    channels = meta.findall('.//Information/Image/Dimensions/Channels/Channel')
    keeplist=[]
    channel_dict = {'channel_indices':[],
                   'channel_names':[]}    
    for index,channel in enumerate(channels):
#         index = channel.attrib['Id'].replace('Channel:','')
        channel_name = channel.attrib['Name']
        channel_dict['channel_indices'].append(str(index))
        channel_dict['channel_names'].append(channel_name)

        try: 
            laser_wavelength = channel.find('.//ExcitationWavelength').text
            laser_intensity = channel.find('.//Intensity').text
        except:
            laser_wavelength = 'None'
            laser_intensity = 'None'

        feats['laser_wavelength'+'_'+channel_name] = laser_wavelength
        feats['laser_intensity'+'_'+channel_name] = laser_intensity
    feats['channel_dict'] = str(channel_dict)

    ZStep = meta.find('.//Z/Positions/Interval/Increment')
    ZStep = eval(ZStep.text)

    ScalingX =  eval(meta.find('.//ScalingX').text)*1e6
    ScalingY =  eval(meta.find('.//ScalingY').text)*1e6
    ScalingZ =  eval(meta.find('.//ScalingZ').text)*1e6
    feats['pixelSizes'] = (ScalingX,ScalingY,ScalingZ)

    feats['imgsize_um'] = tuple([x*y for x,y in zip(feats['pixelSizes'] ,feats['shape'])])

    dfmetalist=[]
    cycle=0
    
    for regions in meta.findall('.//Scenes'):
        for region in regions.findall('.//Scene'):

            attrib = region.attrib
            feats['Name'] = attrib['Index']
            for info in region.findall('.//Position'):
                feats['Y'] = float(info.attrib['X'])
                feats['X'] = float(info.attrib['Y']) #need to switch Y and X to match ZSD coordinates
                feats['Z'] = float(info.attrib['Z'])
            feats['IsUsedForAcquisition'] = 'true'




            dfmetalist.append(pd.DataFrame(data=feats.values(),index=feats.keys()).T)
    df1 = pd.concat(dfmetalist)

    #now search for next set of parameters from .//Scene xml region
    #this pulls out the differences between how scene number and position number!
    dfsp=[]
    info_val_list=[]
    info_attrib_list = ['Position','Scene','Well_id']
    for region in meta.findall('.//Scene'):
        feats={}
        attrib = region.attrib
        feats['Position'] = attrib['Index']
        feats['Position_num'] = int(feats['Position'].replace('P',''))
        feats['Scene'] = int(attrib['Index'])+1
        subregion = region.find('Shape')
        feats['Well_id'] = 'unknown' #in case Shape is not a feature
        for subregion in region.findall('Shape'):
            feats['Well_id'] = subregion.attrib['Name']
        dfsp.append(pd.DataFrame(data = feats.values(), index = feats.keys()).T)

    df2 = pd.concat(dfsp)
    df = pd.merge(df1,df2,left_on='Name',right_on='Position',suffixes=('_1','_2'))
    dfsub =df.copy()
    dfsub = dfsub[dfsub['Scene'].astype(int)<=number_of_scenes_acquired]
    dfsub['fname']=dfsub['file'].apply(lambda x: Path(x).stem)
    return dfsub

def match_to_filenames(df,export_path):
    import os
    import re
    matchlist =[]
    
    files = [x for x in os.listdir(export_path) if bool(re.search('.czi',x))]
    pos = [re.search('P[0-9]+-',x)[0][0:-1] for x in files]
    

    for i in range(df.shape[0]):
        fname = df.iloc[i].at['Name']
        matchstr = [x for x,y in zip(files,pos) if (y==fname)]
        if matchstr:
            matchlist.append(matchstr[0])
        else:
            matchlist.append('missing')

    df['filename'] = matchlist
    return df



def match_to_filenames_dumb(df,export_path):

    
    df['filename'] = ['missing' for x in df['Name'].tolist()]
    return df

def read_position_file(filename):
    pad_tree = ET.parse(filename)
    pad_root = pad_tree.getroot()

    info_val_list=[]
    info_attrib_list = ['Name','X','Y','Z','Id','IsUsedForAcquisition']
    for regions in pad_root.findall('SingleTileRegions'):
        for region in regions.findall('SingleTileRegion'):
            attrib = region.attrib
            Name = attrib['Name']
            Id = str(attrib['Id'])
            for info in region.findall('X'):
                x = float(info.text)
            for info in region.findall('Y'):
                y = float(info.text)
            for info in region.findall('Z'):
                z = float(info.text)
            for info in region.findall('IsUsedForAcquisition'):
                acquire = info.text

            info_val_list.append((Name,x,y,z,Id,acquire))

    
    df = pd.DataFrame(info_val_list,columns=info_attrib_list)
    
    return df

def read_czi_file_for_positions(filename):
    from aicsimageio import AICSImage
    reader = AICSImage(filename)
    meta0 = reader.metadata    
    #convert metadata to lxml
    metastr = ET.tostring(meta0).decode("utf-8")
    meta = etree.fromstring(metastr) 
    pad_root = meta
    
    info_val_list=[]
    info_attrib_list = ['Name','X','Y','Z','Id','IsUsedForAcquisition']
    for regions in pad_root.findall('.//SingleTileRegions'):
        for region in regions.findall('./SingleTileRegion'):
            attrib = region.attrib
            Name = attrib['Name']
            Id = str(attrib['Id'])
            for info in region.findall('X'):
                x = float(info.text)
            for info in region.findall('Y'):
                y = float(info.text)
            for info in region.findall('Z'):
                z = float(info.text)
            for info in region.findall('IsUsedForAcquisition'):
                acquire = info.text

            info_val_list.append((Name,x,y,z,Id,acquire))

    
    df = pd.DataFrame(info_val_list,columns=info_attrib_list)
    
    return df

def make_new_positions_in_tiles(df_orig,mag=100,num_updown=3,num_leftright=3,lrgap=20,udgap=20):
#     lrspacing = 160 #greater than 115 is necessary to ensure that there is no overlap of illumination for 100x
#     udspacing = 115 #greater than 115 is necessary to ensure that there is no overlap of illumination for 100x
    
#     sl_lr = [x*lrspacing for x in list(np.arange(-1*nlr,nlr+1))]
#     sl_ud = [x*udspacing for x in list(np.arange(-1*nud,nud+1))]
#     sl = list(product(sl_ud,sl_lr))

    px = pixel_size_lookup(mag)
#     w=1824*px #units, um width of image
#     h=1248*px #units, um height of image
    
    w=2048*px #units, um width of image
    h=2048*px #units, um height of image
    
    lrspacing = w + lrgap
    udspacing = h + udgap
    
    left_right = np.arange(0,num_leftright,1)
    offset_leftright = list((left_right-np.mean(left_right))*lrspacing)
    up_down = np.arange(0,num_updown,1)
    offset_updown = list((up_down-np.mean(up_down))*udspacing)
    
    offset_list =  list(product(offset_updown,offset_leftright))
    

    id_listnew =  unique_id_list_generator(df_orig,offset_list) #return list of unique ids for czsh file type
    keeplist=[]
    cycle=-1
    for i in range(df_orig.shape[0]):
        X0 = df_orig['X'].iloc[i]
        Y0 = df_orig['Y'].iloc[i]
        Z0 = df_orig['Z'].iloc[i]
        acquire_bool = df_orig['IsUsedForAcquisition'].iloc[i]
        for offset in offset_list:
            lr = offset[1]
            ud = offset[0]
            cycle=cycle+1
            dfdict={}
            dfdict['Name'] = 'P'+str(cycle+1)
            dfdict['X'] = X0+lr
            dfdict['Y'] = Y0+ud
            dfdict['Z'] = Z0 #adjust Z in a separate step if desired
            dfdict['Id'] = id_listnew[cycle]
            dfdict['IsUsedForAcquisition'] = acquire_bool
            keeplist.append(dfdict.values())

    dfnew = pd.DataFrame(keeplist,columns=dfdict.keys())
    return dfnew
    

def adjust_z(df,zoffset):
    zlist = df['Z'].to_numpy()
    zlist = zlist + zoffset  #zoffset is in units of um
    df['Z'] = zlist
    return df

def adjust_xy(df,args_dict):
    keylist = list(args_dict.keys())
    xykeylist = [x for x in keylist if bool(re.search('X|Y',x,re.IGNORECASE))]
    if len(xykeylist)>0:
        for xykey in xykeylist:
            xyval = float(args_dict[xykey])
            xylist = df[xykey.capitalize()].to_numpy()
            xylist = xylist + xyval #xyval is offset in units of um
            df[xykey] = xylist
    return df


def write_dataframe_into_xmlold(filename,df):
    pad_tree = ET.parse(filename)
    pad_root = pad_tree.getroot()
    
    #first remove all imaging regions/tiles/positions
    pad_list = [pad_regions.tag for pad_regions in pad_tree.findall('*')]
    pad_list = [x for x in pad_list if bool(re.search('Tileregions|Tileregionarrays',x,re.IGNORECASE))]
    for pad_item in pad_list:
        for pad_regions in pad_tree.findall('./'+pad_item):
            for region in pad_regions.findall('*'):
                pad_regions.remove(region)

    #now write the desired tile regions into the calibrated plate xmltree
    for i in range(df.shape[0]):
        X = df['X'].iloc[i]
        Y = df['Y'].iloc[i]
        Z = df['Z'].iloc[i]
        Id = df['Id'].iloc[i]
        Name = df['Name'].iloc[i]
        acquire = df['IsUsedForAcquisition'].iloc[i]

        attrib = { "Id":Id,"Name": Name}
        b = ET.Element('SingleTileRegion',attrib=attrib)
        #now fill in details of b with subelements before appending b to list
        c = ET.SubElement(b, 'X')
        c.text = str(X)
        d = ET.SubElement(b, 'Y')
        d.text = str(Y)
        e = ET.SubElement(b, 'Z')
        e.text = str(Z)
        f = ET.SubElement(b,'IsUsedForAcquisition')
        f.text = acquire


        pad_regions.append(b)
    return pad_tree,pad_root

def write_dataframe_into_xml(filename,df):
    pad_tree = ET.parse(filename)
    pad_root = pad_tree.getroot()
    
    #first remove all imaging regions/tiles/positions
    pad_list = [pad_regions.tag for pad_regions in pad_tree.findall('*')]
    pad_list = [x for x in pad_list if bool(re.search('Tileregions|Tileregionarrays',x,re.IGNORECASE))]
    for pad_item in pad_list:
        for pad_regions in pad_tree.findall('./'+pad_item):
            for region in pad_regions.findall('*'):
                pad_regions.remove(region)

    singletileregion = pad_tree.find('./SingleTileRegions')
    #now write the desired tile regions into the calibrated plate xmltree
    for i in range(df.shape[0]):
        X = df['X'].iloc[i]
        Y = df['Y'].iloc[i]
        Z = df['Z'].iloc[i]
        Id = df['Id'].iloc[i]
        Name = df['Name'].iloc[i]
        acquire = df['IsUsedForAcquisition'].iloc[i]

        attrib = { "Id":Id,"Name": Name}
        b = ET.Element('SingleTileRegion',attrib=attrib)
#         b = ET.SubElement(singletileregion,'SingleTileRegion',attrib=attrib)
        #now fill in details of b with subelements before appending b to list
        c = ET.SubElement(b, 'X')
        c.text = str(X)
        d = ET.SubElement(b, 'Y')
        d.text = str(Y)
        e = ET.SubElement(b, 'Z')
        e.text = str(Z)
        f = ET.SubElement(b,'IsUsedForAcquisition')
        f.text = acquire


        singletileregion.append(b)
        
    return pad_tree,pad_root



def write_dataframe_into_xml_any(metadata_from_file,df):
    pad_root = metadata_from_file
    pad_tree = metadata_from_file
    
    #first remove all imaging regions/tiles/positions
    pad_list = [pad_regions.tag for pad_regions in pad_tree.findall('*')]
    pad_list = [x for x in pad_list if bool(re.search('Tileregions|Tileregionarrays',x,re.IGNORECASE))]
    for pad_item in pad_list:
        for pad_regions in pad_tree.findall('./'+pad_item):
            for region in pad_regions.findall('*'):
                pad_regions.remove(region)

    singletileregion = pad_tree.find('.//SingleTileRegions')
    #now write the desired tile regions into the calibrated plate xmltree
    for i in range(df.shape[0]):
        X = df['X'].iloc[i]
        Y = df['Y'].iloc[i]
        Z = df['Z'].iloc[i]
        Id = df['Id'].iloc[i]
        Name = df['Name'].iloc[i]
        acquire = df['IsUsedForAcquisition'].iloc[i]

        attrib = { "Id":Id,"Name": Name}
        b = ET.Element('SingleTileRegion',attrib=attrib)
#         b = ET.SubElement(singletileregion,'SingleTileRegion',attrib=attrib)
        #now fill in details of b with subelements before appending b to list
        c = ET.SubElement(b, 'X')
        c.text = str(X)
        d = ET.SubElement(b, 'Y')
        d.text = str(Y)
        e = ET.SubElement(b, 'Z')
        e.text = str(Z)
        f = ET.SubElement(b,'IsUsedForAcquisition')
        f.text = acquire


        singletileregion.append(b)
        
    return pad_tree,pad_root


def write_zen_positionfile(old_filename,xml_tree,xml_root,mag1,mag2,extrastr=''):

    filename = old_filename.replace(mag1,'').replace('.czsh',mag2+'_created.czsh')

    savepath = Path(os.path.dirname(old_filename))/ str(os.path.basename(filename))
    indent(xml_root)
    xml_tree.write(savepath, encoding="utf-8",xml_declaration=True,default_namespace=None, method=None, short_empty_elements=False,)
    print(savepath)

    #this block of code ensures the first line is correctly written to begin with the unicode character '\ufeff'    
    # savepathold = Path(os.path.abspath(os.curdir))/ '20200225_F02_positions.czsh'
    savepathold = Path(old_filename)

    with open(savepathold, encoding='utf-8', mode='r') as txtFile:
        line_list = txtFile.readlines()
        txtFile.close()

    with open(savepath, encoding='utf-8', mode='r') as txtFile:
        line_list = txtFile.readlines()
        line_edit = '\ufeff' + line_list[0]
        line_list[0] = line_edit.replace("'",'"')
        line_list[-1]=line_list[-1][0:-1]
        txtFile.close()

    line_list[-1] = line_list[-1] +'\n'
    finalsavepath=str(savepath).replace('.czsh','_forimport'+extrastr+'.czsh')
    with open(finalsavepath, encoding='utf-8', mode='w+') as txtFile:
        for line in line_list:
            txtFile.write(line)
        txtFile.close()
    print("file saved as - ",finalsavepath)
    return finalsavepath

def write_zen_positionfile_fromczi(old_filename,xml_tree,xml_root,mag1,mag2,extrastr=''):

    filename = old_filename.replace(mag1,'').replace('.czi',mag2+'_created.czsh')

    savepath = Path(os.path.dirname(old_filename))/ str(os.path.basename(filename))
    indent(xml_root)
    
    xml_tree.write(savepath, encoding="utf-8",xml_declaration=True,default_namespace=None, method=None, short_empty_elements=False,)
    print(savepath)

    #this block of code ensures the first line is correctly written to begin with the unicode character '\ufeff'    
    # savepathold = Path(os.path.abspath(os.curdir))/ '20200225_F02_positions.czsh'
    savepathold = Path(old_filename)

    with open(savepathold, encoding='utf-8', mode='r') as txtFile:
        line_list = txtFile.readlines()
        txtFile.close()

    with open(savepath, encoding='utf-8', mode='r') as txtFile:
        line_list = txtFile.readlines()
        line_edit = '\ufeff' + line_list[0]
        line_list[0] = line_edit.replace("'",'"')
        line_list[-1]=line_list[-1][0:-1]
        txtFile.close()

    line_list[-1] = line_list[-1] +'\n'
    finalsavepath=str(savepath).replace('.czsh','_forimport'+extrastr+'.czsh')
    with open(finalsavepath, encoding='utf-8', mode='w+') as txtFile:
        for line in line_list:
            txtFile.write(line)
        txtFile.close()
    print("file saved as - ",finalsavepath)
    return finalsavepath


def pixel_size_lookup(mag):
    #returns unbinned pixel sizes
    if mag=='20x':
        px = 0.271
    elif mag=='63x':
        px = 0.0857
    elif mag=='100x':
        px = 0.054
    return px
    
    
    
def create_image_of_tiling(filename,df1,df2,mag1,mag2,num_updown,num_leftright):
    import matplotlib.pyplot as plt
    #make a scatter plot of positions
    npt = 1 #number of 20x positions to examine
    X = df1['X'].to_numpy()
    Y = df1['Y'].to_numpy()
    X = X[0:npt].copy()
    Y = Y[0:npt].copy()
    px1 = pixel_size_lookup(mag1)
    w=1848*px1
    h=1248*px1
    cycle=-1
    for xx,y in zip(X,Y):
        plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor='y',alpha=0.5)
        cycle=cycle+1
    #     plt.text(xx,y,str(cycle))

    X = df2['X'].to_numpy()
    Y = df2['Y'].to_numpy()
    
    nph = npt*num_updown*num_leftright
    X = X[0:nph].copy()
    Y = Y[0:nph].copy()
    px2 = pixel_size_lookup(mag2)
    w=1848*px2
    h=1248*px2
    cycle=-1
    for xx,y in zip(X,Y):
        plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor='r',alpha=0.5)
        cycle=cycle+1
        plt.text(xx,y,str(cycle+1))
        
    w=2048*px2
    h=2048*px2
    cycle=-1
    for xx,y in zip(X,Y):
        plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor='r',alpha=0.1,edgecolor='k')
        cycle=cycle+1
        plt.text(xx,y,str(cycle+1))
        
    imgsavename = filename.replace('czsh','png')
    plt.savefig(imgsavename,format='png',dpi=300)
    plt.show()
    print('imagesavedas ',imgsavename)


    
def plot_matched_positions(dfmatch,suffixes=('_mag1','_mag2')):
    import matplotlib.pyplot as plt
    dfmatch['mag1_position_number'+suffixes[0]] = dfmatch['Position'+suffixes[0]].apply(lambda x: int(x[1::]))
    for well,dfm in dfmatch.groupby('Well_id'+suffixes[0]):
        plt.figure(figsize=(10,10))
        plt.title(well)
        for mag1pos,dfsub in dfm.groupby(['mag1_position_number'+suffixes[0]]):
            try:
                X = dfsub['X_refmatched'+suffixes[0]].to_numpy()[0:1]
                Y = dfsub['Y_refmatched'+suffixes[0]].to_numpy()[0:1]
            except:
                X = dfsub['X'+suffixes[0]].to_numpy()[0:1]
                Y = dfsub['Y'+suffixes[0]].to_numpy()[0:1]
            mag1scene = dfsub['Scene'+suffixes[0]].tolist()[0:1]

            imgsize = dfsub.iloc[0].at['imgsize_um'+suffixes[0]]
            w = imgsize[0]
            h = imgsize[1]

            cycle=-1

            for xx,y in zip(X,Y):
                plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor='y',alpha=0.5)
                cycle=cycle+1
                plt.text(xx-w/2,y+h/2,str(mag1pos)+', ' +str(mag1scene))
                

            imgsize2 = dfsub.iloc[0].at['imgsize_um'+suffixes[1]]
            w = imgsize2[0]
            h = imgsize2[1]
            for i,(mag2pos,dfsubsub) in enumerate(dfsub.groupby(['Position'+suffixes[1]])):
                try:
                    xx = dfsubsub['X_refmatched'+suffixes[1]].to_numpy()[0]
                    y = dfsubsub['Y_refmatched'+suffixes[1]].to_numpy()[0]
                except:
                    xx = dfsubsub['X'+suffixes[1]].to_numpy()[0]
                    y = dfsubsub['Y'+suffixes[1]].to_numpy()[0]
                plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor='r',alpha=0.5)
                plt.text(xx,y,str(mag2pos),ha='center')

                plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor='r',alpha=0.1,edgecolor='k')

        plt.axis('square')
        plt.show()

        
        
def plot_all_positions(dfmag1,dfmag2,adjust=False):
    import matplotlib.pyplot as plt
    
    
    if adjust:
        adjuststr = '_adjusted'
    else:
        adjuststr = ''
    
    plt.figure(figsize=(10,10))
    well_list = dfmag1.Well_id.unique()
    fig,axr = plt.subplots(nrows=len(well_list),ncols=1,figsize=(10,10*len(well_list)))
    try:
        axlist = axr.reshape(-1,)
    except:
        axlist = [axr]
    for di,(df1,cc) in enumerate(zip([dfmag1,dfmag2],['y','r'])):
        df1['mag1_position_number'] = df1['Position'].apply(lambda x: int(x[1::]))
        for wi,(well,dfm) in enumerate(df1.groupby('Well_id')):
            plt.sca(axlist[wi])
            plt.title(well)
            for mag1pos,dfsub in dfm.groupby(['mag1_position_number']):
                X = dfsub['X'+adjuststr].to_numpy()[0:1]
                Y = dfsub['Y'+adjuststr].to_numpy()[0:1]
                mag1scene = dfsub['Scene'].tolist()[0:1]

                imgsize = dfsub.iloc[0].at['imgsize_um']
                w = imgsize[0]
                h = imgsize[1]

                cycle=-1

                for xx,y in zip(X,Y):
                    if di>0:
                        hatch='//'
                    plt.fill_between(x=(xx-w/2,xx+w/2),y1=(y-h/2,y-h/2),y2=(y+h/2,y+h/2),facecolor=cc,alpha=0.5,hatch=hatch)
                    cycle=cycle+1
                    if cc=='y':
#                         plt.text(xx-w/2,y+h/2,str(mag1pos)+', ' +str(mag1scene))
                        plt.text(xx-w/2,y+h/2,str(mag1scene[0]))


    plt.axis('square')
    plt.show()
    
def generate_match(dfmag1,dfmag2,suffixes=('_mag1','_mag2')):
    dfmatch = pd.merge(dfmag1,dfmag2,on='mag1_position_number',suffixes=suffixes)
#     dfmatch = dfmatch[dfmatch['IsUsedForAcquisition'+suffixes[1]]=='true']
#     dfmatch = dfmatch[dfmatch['IsUsedForAcquisition'+suffixes[0]]=='true']
    dfmatch = dfmatch[dfmatch['elist']==False]
    
    #choose specific columns
    columns = list(dfmatch.columns)
    cc = ['filename'+suffixes[0],'filename'+suffixes[1]]
    chosen_columns = ['imgsize','X','Y','Z','X_refmatched','Y_refmatched','Z_refmatched','pixelSizes','parent_file','Well_id','Position','Scene','totalmagnification','laser_wavelength','exposure_time','laser_intensity','channel_dict',suffixes[0].replace('_','')+'_position_number'+suffixes[0]]
    choose_columns = [x for x in columns if bool(re.search('|'.join(chosen_columns),x))]
    cc.extend(choose_columns)
    dfmatch = dfmatch[cc]
    
    
    #find offset from mag1
    xy_offset_from_mag1_center=[]
    for i in range(dfmatch.shape[0]):
        xyz20 = tuple(list(dfmatch[['X'+suffixes[0],'Y'+suffixes[0],'Z'+suffixes[0]]].iloc[i]))
        xyz100 = tuple(list(dfmatch[['X'+suffixes[1],'Y'+suffixes[1],'Z'+suffixes[1]]].iloc[i]))
        offset = np.asarray(xyz20)-np.asarray(xyz100)
        xy_offset_from_mag1_center.append(tuple(offset[0:2]))
    try:
        dfmatch.insert(2,column=suffixes[1].replace('_','')+'_xy_offset_from'+suffixes[0]+'_center',value=xy_offset_from_mag1_center,allow_duplicates=False)
    except:
        print("already added")
    
    
    return dfmatch
    

def unique_id_list_generator(df_twentyX,sl):
    import string
    import random
    #example of id  = 637139503441604687
    # 6371395 is always present at the beginning of the id

    N=len('03441604687') #standard id length
    id_listnew=[]

    cycle_final = df_twentyX.shape[0]*len(sl)
    li = len(id_listnew)
    while li<cycle_final:
        id_listnew.append('6371395'+''.join(random.choice(string.digits) for _ in range(N)))
        id_listnew = list(set(id_listnew))
        li = len(id_listnew)
    return id_listnew

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            
         
        
def write880_posfile(df,txtFilePath):
    import pandas as pd
    import os

#     root = tk.Tk()
#     root.withdraw()
#     csvPath = filedialog.askopenfilename(filetypes=(("csv files", "*.csv"),))
#     #txtDir = os.path.dirname(csvPath)
#     #txtFilename = csvPath.split('/')[-1].replace('.csv','.txt')
#     txtFilePath = csvPath.replace('.csv','.txt')
#     df = pd.read_csv(csvPath)

    print(df)
    Xnew = df['X'].values
    Ynew = df['Y'].values
    Znew = df['Z'].values
    numPos = len(Xnew)

   
    txtFile = open(txtFilePath,'w')
    txtFile.write('Carl Zeiss LSM 510 - Position list file - Version = 1.000\n')
    txtFile.write('BEGIN PositionList Version = 10001\n')
    txtFile.write('\tBEGIN  10001\n')
    txtFile.write('\t\tRelativePositions = 1\n')
    txtFile.write('\t\tReferenceX = -0.000 µm\n')
    txtFile.write('\t\tReferenceY = -0.000 µm\n')
    txtFile.write('\t\tReferenceZ = -0.000 µm\n')
    txtFile.write('\tEND\n')
    txtFile.write(f'\tNumberPositions = {numPos}\n')

    for i in range(numPos):

        txtFile.write(f'\tBEGIN Position{i+1} Version = 10001\n')
        txtFile.write('\t\tX = {0:.2f} µm\n'.format(Xnew[i]))
        txtFile.write('\t\tY = {0:.2f} µm\n'.format(Ynew[i]))
        txtFile.write(f'\t\tZ = {int(Znew[i])} µm\n')
        txtFile.write('\tEND\n')

    txtFile.write('END')
    txtFile.write('\n')
    txtFile.close()