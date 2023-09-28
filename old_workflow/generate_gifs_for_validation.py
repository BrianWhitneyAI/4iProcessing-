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

parser = argparse.ArgumentParser()
parser.add_argument("--input_yaml", type=str, required=True, help="yaml config path")
parser.add_argument('--frame_rate', type=int, default=500)
parser.add_argument("--placeholder", type=str, required=False, help="placeholder for snakemake rule.... generates an output text file")

def normalization(img):
    #rescaled_img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    #img_stack_list_rs = []
    #for img in img_stack_list:
    lp = np.nanpercentile(img,1)
    hp = np.nanpercentile(img,90)
    img_rs = skex.rescale_intensity(img,in_range=(lp,hp),out_range='uint8').astype('uint8')
    #img_stack_list_rs.append(img_rs)
    return img_rs

def min_max_norm(img):
    rescaled_img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    return rescaled_img

def generate_gif_for_evaluation(input_dir, filenames, frame_rate, output_dir, position, barcode):
    imglist=[]
    filenames.sort()
    #print("filenames are of size {}".format(len(filenames)))
    print(filenames)
    for i in range(len(filenames)):

        img_np = tifffile.imread(os.path.join(input_dir, filenames[i]))
        img_rescaled = normalization(img_np)        

        img = Image.fromarray(img_rescaled)
        
        img_resized = img.resize(gif_image_shape)
        img_w_text = ImageDraw.Draw(img_resized)
        img_w_text.text((40, 40), os.path.basename(filenames[i]).split("-Scene",1)[0], fill=(255))
        #print(np.shape(img_resized))
        #img_resized.save("sample.png")
        imglist.append(img_resized.convert('P'))

    imglist[0].save(os.path.join(output_dir, f'{barcode}_position_{str(position).zfill(2)}_evaluation.gif'),
               save_all=True, append_images=imglist[1:], optimize=False, duration=frame_rate, loop=0)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.input_yaml) as f:
        yaml_config = yaml.load(f, Loader=SafeLoader)    

    output_aligned_images_dir = os.path.join(yaml_config["output_path"], str(yaml_config["barcode"]), "round_aligned_images")

    output_gif_save_dir = os.path.join(yaml_config["output_path"], str(yaml_config["barcode"]), "validation_gifs")
    if not os.path.exists(output_gif_save_dir):
        os.mkdir(output_gif_save_dir)

    gif_image_shape = (int(1847/4), int(1247/4)) # have this be automatically figured out from shape dims of FOV

    max_tp = np.max([int(f.split("T", 1)[1].split("_C", 1)[0]) for f in os.listdir(output_aligned_images_dir) if "R0" in f and "C2" in f])

    filenames = [f for f in os.listdir(output_aligned_images_dir) if "R0" in f and "C2" in f and f"T{max_tp}" in f]

    timelapse_ref_channel = 2
    round_ref_channel = 3

    for i in range(len(filenames)):
        filenames_for_position = []
        # filenames_for_position.append(filenames[i])
        #round= filenames[i].split("_R", 1)[1].split("_P", 1)[0]
        position = filenames[i].split("_P", 1)[1].split("-", 1)[0]
        # import pdb
        # pdb.set_trace()
        filenames_in_rounds_for_position = [x for x in os.listdir(output_aligned_images_dir) if f"P{position}" in x and "C3" in x and f"R{0}" not in x]
        filenames_in_rounds_for_position.append(filenames[i])


        if len(filenames_in_rounds_for_position) !=0:
            generate_gif_for_evaluation(output_aligned_images_dir, filenames_in_rounds_for_position, args.frame_rate, output_gif_save_dir, position, yaml_config["barcode"])
            #print(filenames_for_position)

    if args.placeholder:
        f = open(args.placeholder, "a")
        f.write("done")
        f.close()
