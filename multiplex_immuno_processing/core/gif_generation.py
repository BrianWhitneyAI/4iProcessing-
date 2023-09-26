import os
import numpy as np
import glob
import re
import tifffile
from PIL import Image, ImageDraw
import argparse
from skimage import exposure
import skimage.exposure as skex
from yaml.loader import SafeLoader
import ruamel.yaml
import yaml




def normalization_gif(img):
    #rescaled_img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    #img_stack_list_rs = []
    #for img in img_stack_list:
    lp = np.nanpercentile(img,1)
    hp = np.nanpercentile(img,90)
    img_rs = skex.rescale_intensity(img,in_range=(lp,hp),out_range='uint8').astype('uint8')
    #img_stack_list_rs.append(img_rs)
    return img_rs

def generate_gif_for_evaluation(input_dir, filenames, output_dir, position, barcode, gif_image_shape, frame_rate=500):
    """
    Takes in list of input filenames, frame rate(milliseconds), output_dir, position, and barcode and generates a gif 
    that iterates through each round in the position
    """
    imglist=[]
    filenames.sort()
    #print("filenames are of size {}".format(len(filenames)))
    print(filenames)
    for i in range(len(filenames)):

        img_np = tifffile.imread(os.path.join(input_dir, filenames[i]))
        img_rescaled = normalization_gif(img_np)        

        img = Image.fromarray(img_rescaled)

        
        img_resized = img.resize(gif_image_shape)
        img_w_text = ImageDraw.Draw(img_resized)
        img_w_text.text((40, 40), os.path.basename(filenames[i]).split("-Scene",1)[0], fill=(255))
        #print(np.shape(img_resized))
        #img_resized.save("sample.png")
        imglist.append(img_resized.convert('P'))

    imglist[0].save(os.path.join(output_dir, f'{barcode}_position_{str(position).zfill(2)}_evaluation.gif'),
               save_all=True, append_images=imglist[1:], optimize=False, duration=frame_rate, loop=0)
