import numpy as np
from aicsimageio import AICSImage
import re
from czi_helper import czi_metadata_helper



def max_project(seg_img_labeled):
    xy_seg_maxproj = np.max(seg_img_labeled, axis=0)[np.newaxis, ...][0,:,:]
    return xy_seg_maxproj

def load_zstack_mip(filepath, refrence_channel, scene, timepoint):
    reader = AICSImage(filepath)
    reader.set_scene(int(scene-1)) # b/c of zero indexing ---- this is not reflected in ZEN GUI
    img = reader.data[timepoint, refrence_channel, :, :, :] # getting T, ch, Z, Y, X
    
    return max_project(img)


def get_FOV_shape(filepath):
    reader = AICSImage(filepath)
    return np.shape(reader)




def get_round_info_from_dict(round_of_intrest, dataset):
    """Returns the information for the round of intrest from a dict that is structured according to our yaml config file"""
    round_info= [dataset['Data'][f] for f in range(len(dataset['Data'])) if dataset['Data'][f]['round'] == round_of_intrest]
    assert len(round_info)==1, "Multiple rounds with the same name found or none found"
    return round_info[0]

def get_available_positions(round_info):
    """loads the czi and ckecks the list of positions available"""
    #dataframe = zen_position_helper.get_position_info_from_czi(round_info['path'])
    czi_file = czi_metadata_helper(round_info['path'])
    positions, scenes, well_ids = czi_file.get_position_scene_wellid_paired_list()
    assert len(positions) == len(scenes) == len(well_ids)
    return positions, scenes, well_ids

def find_matching_position_scene(all_positions_in_round, all_scenes_in_round, well_ids_all, refrence_postion):
    """returns the corresponding position, scene pair that matches refrence position in refrence round"""
    position_index = all_positions_in_round.index(refrence_postion)
    matching_scene = all_scenes_in_round[position_index]
    well_id_match = well_ids_all[position_index]

    return all_positions_in_round[position_index], matching_scene, well_id_match
