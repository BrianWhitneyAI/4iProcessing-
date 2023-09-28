import os
import argparse
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from aicsimageio import AICSImage
import numpy as np
import tifffile
from scipy.ndimage import affine_transform
import ast
import re
from core.gif_generation import generate_gif_for_evaluation
from core.registration_utils import get_shift_to_center_matrix_2D, get_align_matrix_2D
from core.utils import load_zstack_mip, get_FOV_shape, max_project

"""
perform the alignment from the csvs 
"""


parser = argparse.ArgumentParser()
parser.add_argument("--input_yaml", type=str, required=True, help="yaml config path")
parser.add_argument("--matched_position_w_align_params_csv", type=str, required=False, help="Matched position csv to align a single position (Optional)")
parser.add_argument("--round_crop_tempelate", type=str, default="Timelapse")
parser.add_argument("--placeholder", type=str, required=False, help="placeholder for snakemake rule.... generates an output text file")



def find_files_to_use_in_gif(filenames_all):
    "This is currently setup this way b/c of channel switching in the data"
    Round_imaging_nuc_images = [f for f in filenames_all if "_C3_" in f and "_R0_" not in f]
    Round0_nuc_images = [f for f in filenames_all if "_C2_" in f and "_R0_" in f]
    if len(Round0_nuc_images)==1:
        Round_imaging_nuc_images.append(Round0_nuc_images[0])
    
    return Round_imaging_nuc_images


class perform_alignment_per_position():
    def __init__(self, alignment_csv_dir, yaml_config, round_to_crop_to, generate_validation_gif=True):
        df = pd.read_csv(alignment_csv_dir)
        cond = df["Round"]==round_to_crop_to
        matching_indices = df[cond].index
        matching_rows = df.loc[matching_indices]
        remaining_rows = df.loc[~cond]
        self.position_csv = pd.concat([matching_rows, remaining_rows]) # makes sure 
        
        with open(yaml_config) as f:
            self.yaml_config = yaml.load(f, Loader=SafeLoader)

        assert os.path.exists(os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"]))), "output dir does not exist"

        self.round_to_crop_tempelate = round_to_crop_to
        self.position = os.path.basename(alignment_csv_dir)

        self.save_aligned_images_dir = os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"]),"round_aligned_images")
        

        if not os.path.exists(self.save_aligned_images_dir):
            os.mkdir(self.save_aligned_images_dir)
        
        self.generate_validation_gif=generate_validation_gif
        if self.generate_validation_gif==True: 
            self.output_gif_save_dir = os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"]), "validation_gifs")
            if not os.path.exists(self.output_gif_save_dir):
                os.mkdir(self.output_gif_save_dir)

    def get_tempelate_ref(self, y_dim, x_dim):
        tempelate_ref = np.uint16(np.asarray([y_dim+ (y_dim * 0.33), x_dim + (x_dim * 0.33),]))
        return tempelate_ref
    

    def registeration_using_alignment_params(self, raw_mip, tempelate_ref, alignment_offset):
        """Construct homography matrix and do alignment"""
        shift_to_center_matrix = get_shift_to_center_matrix_2D(raw_mip.shape, tempelate_ref)
        align_matrix = get_align_matrix_2D(alignment_offset)
        combo = shift_to_center_matrix @ align_matrix # matrix multiplication
        aligned_mip = affine_transform(raw_mip, np.linalg.inv(combo), output_shape=tempelate_ref, order=0)
        return aligned_mip


    def find_padding_dimensions(self, padded_img):
        nonzero_rows, nonzero_cols = np.nonzero(padded_img)
        min_y, max_y = min(nonzero_rows), max(nonzero_rows)
        min_x, max_x = min(nonzero_cols), max(nonzero_cols)
        return (min_y, max_y, min_x, max_x)

    def crop_according_to_refrence_crop_round(self, aligned_mip, crop_dims):

        cropped_img = aligned_mip[crop_dims[0]: crop_dims[1]+1, crop_dims[2]: crop_dims[3]+1]

        return cropped_img

    def save_mip(self, image, round, position, well_id, channel, timepoint):
        # save the mip
        round_num = re.findall(r'\d+', round)
        if round_num == []:
            round_num = 0
        else:
            round_num = round_num[0]
        barcode = self.yaml_config["barcode"]


        name = f"{barcode}_R{round_num}_P{str(position).zfill(2)}-{well_id}-T{str(timepoint).zfill(2)}_C{channel}_MIP.tiff"
        tifffile.imwrite(os.path.join(self.save_aligned_images_dir,name), image)
        return name


    def perform_alignment(self):
        """Alignment for specified position"""
        _, _,_, y_dim_ref,x_dim_ref = get_FOV_shape(self.position_csv.iloc[0]["RAW_filepath"])
        tempelate_ref = self.get_tempelate_ref(y_dim_ref, x_dim_ref)

        filenames_in_rounds_for_position=[]
        for i in range(np.shape(self.position_csv)[0]):

            round_of_intrest = self.position_csv.iloc[i]
            t_dim, ch_dim, z_dim, y_dim, x_dim = get_FOV_shape(round_of_intrest["RAW_filepath"])
            
            for timepoint in range(t_dim):
                for ch in range(ch_dim):
                    mip_to_align = load_zstack_mip(round_of_intrest["RAW_filepath"], ch, round_of_intrest["Scene"], timepoint)
                    params_to_align_all = ast.literal_eval(round_of_intrest["cross_cor_params"])
                    params_to_align_yx = [params_to_align_all[1], params_to_align_all[2]]
                    aligned_mip = self.registeration_using_alignment_params(mip_to_align, tempelate_ref, params_to_align_yx)

                    if i==0 and ch==0 and timepoint==0:
                        crop_dims = self.find_padding_dimensions(aligned_mip)

                    final_mip = self.crop_according_to_refrence_crop_round(aligned_mip, crop_dims)
                    filename = self.save_mip(final_mip, round_of_intrest["Round"], round_of_intrest["Position"], round_of_intrest["Well_id"], ch, timepoint)
                    

                    if timepoint==t_dim-1: # Aligns last timepoint in R0
                        filenames_in_rounds_for_position.append(filename)


        filenames_in_rounds_for_position = find_files_to_use_in_gif(filenames_in_rounds_for_position)
        if self.generate_validation_gif == True and len(filenames_in_rounds_for_position) !=0 and "Timelapse" in self.position_csv["Round"].tolist():
            # only run contact sheet generation for files that have timelapse round present
            gif_image_shape = (int(x_dim_ref/4), int(y_dim_ref/4))
            generate_gif_for_evaluation(self.save_aligned_images_dir, filenames_in_rounds_for_position, self.output_gif_save_dir, round_of_intrest["Position"], yaml_config["barcode"], gif_image_shape)

if __name__ == "__main__":

    args=parser.parse_args()


    with open(args.input_yaml) as f:
        yaml_config = yaml.load(f, Loader=SafeLoader)

    if args.matched_position_w_align_params_csv:
        position_aligner = perform_alignment_per_position(args.matched_position_w_align_params_csv, args.input_yaml, args.round_crop_tempelate)
        position_aligner.perform_alignment()

    else:
        alignment_parameters_dir = os.path.join(yaml_config["output_path"], str(yaml_config["barcode"]), "alignment_parameters")
        assert os.path.exists(alignment_parameters_dir), "alignment_parameters_dir doesn't exist"
        filenames = [f for f in os.listdir(alignment_parameters_dir) if f.endswith(".csv") and not f.startswith(".")]
        for filename in filenames:
            print(os.path.join(alignment_parameters_dir, filename))
            position_aligner = perform_alignment_per_position(os.path.join(alignment_parameters_dir, filename), args.input_yaml, args.round_crop_tempelate)
            position_aligner.perform_alignment()

    print(f"args placeholder is {args.placeholder}")

    if args.placeholder:
        print("inside")
        f = open(args.placeholder, "a")
        f.write("done")
        f.close()













