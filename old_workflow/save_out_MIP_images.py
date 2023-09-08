from PIL import Image
import tifffile
from aicsimageio import AICSImage
import os
from aicsimageio.writers import OmeTiffWriter
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir_movie", type=str, default="/allen/aics/microscopy/Data/RnD_Sandbox/Timelapse at different timepoint 20221220/5500000123_R00_Chir_44-30.czi")
parser.add_argument("--output_dir", type=str, default="/allen/aics/microscopy/Data/RnD_Sandbox/Timelapse at different timepoint 20221220/output")






def get_max_proj(img):
    '''returns max project of image'''
    max_proj = np.max(img, axis=0)[np.newaxis, ...][0,:,:]
    return max_proj


if __name__ == "__main__":
    args= parser.parse_args()


    reader = AICSImage(args.input_dir_movie)
    num_scenes=40
    num_timepoints=reader.shape[0]
    num_channels=reader.shape[1]
    # output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir_movie).split(".czi", 1)[0])

    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    for scene in range(num_scenes):
        reader.set_scene(scene)
        for timepoint in range(num_timepoints):
            for channel in range(num_channels):
                img = reader.get_image_dask_data("ZYX", C = channel, T = timepoint)
                prefix = "Chir" + os.path.basename(args.input_dir_movie).split("Chir", 1)[1].split(".czi", 1)[0]
                out_fn = os.path.join(args.output_dir, f"{prefix}_P{scene:02d}_T{timepoint:02d}_Ch{channel}_mip.tiff")
                mip_img = get_max_proj(img.compute())
                OmeTiffWriter.save(mip_img, out_fn)
                print(f"saved {out_fn}")

                
                




