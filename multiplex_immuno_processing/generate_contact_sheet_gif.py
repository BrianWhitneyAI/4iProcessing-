import os
import numpy as np
import glob
import re
import tifffile
from PIL import Image, ImageDraw
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--barcode', type=int, required=True)
parser.add_argument('--frame_rate', type=int, default=500)


def normalization(img):
    rescaled_img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    return rescaled_img


def generate_gif_for_evaluation(input_dir, filenames, frame_rate, output_dir, position, barcode):
    imglist=[]
    filenames.sort()
    print("filenames are of size {}".format(len(filenames)))
    for i in range(len(filenames)):
        print("R"+str(i+1).zfill(2))
        #TODO: go from round1-round11 in order -- simple regex
        #filename=[f for f in filenames_for_position if "R"+str(i+1).zfill(2) in f]
        #assert len(filename)==1, "regex failed"
        #filename_round = filename[0]

        img_np = tifffile.imread(os.path.join(input_dir,filenames[i]))
        img_rescaled = normalization(img_np)
        img = Image.fromarray(img_rescaled)
        
        img_resized = img.resize(gif_image_shape)
        img_w_text = ImageDraw.Draw(img_resized)
        # 1664, 2464
        img_w_text.text((20, 20), os.path.basename(filenames[i]).split("-Scene",1)[0], fill=(255))
        print(np.shape(img_resized))
        #img_resized.save("sample.png")
        imglist.append(img_resized.convert('P'))

    imglist[0].save(os.path.join(output_dir, f'{barcode}_position_{str(position).zfill(2)}_evaluation.gif'),
               save_all=True, append_images=imglist[1:], optimize=False, duration=frame_rate, loop=0)
    




if __name__ == "__main__":
    args = parser.parse_args()
    filenames = [f for f in os.listdir(args.input_dir) if "-c04_" in f and f.endswith(".tif")] # channel 4 is the nuclei channel
    output_gif_eval_dir = os.path.join(args.output_dir, f"{args.barcode}_evaluation")
    if not os.path.exists(output_gif_eval_dir):
        os.mkdir(output_gif_eval_dir)

    #print(filenames[0])
    gif_image_shape = (int(2464/4), int(1664/4))
    # for each position in this, load up the image and generate a seperate gif
    for i in range(40): # 40 total positions contained-- usually TODO: regex scheming to find the max position 
        expression_re = "_P" + str(i).zfill(2) + "-"
        #print(expression_re)
        filenames_for_position = [f for f in filenames if expression_re in f]
        print(len(filenames_for_position))
        if len(filenames_for_position) !=0:
            generate_gif_for_evaluation(args.input_dir, filenames_for_position, args.frame_rate, output_gif_eval_dir, i, args.barcode)
            #print(filenames_for_position)
        