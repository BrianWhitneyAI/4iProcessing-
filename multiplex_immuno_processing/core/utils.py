import numpy as np
from aicsimageio import AICSImage





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