import os
import argparse
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from aicsimageio import AICSImage
import numpy as np
import registration_utils
import tifffile
from scipy.ndimage import affine_transform
import ast
"""
perform the alignment from the csvs 

"""

parser = argparse.ArgumentParser()
parser.add_argument("--input_matched_position_csv_dir", type=str, default="/allen/aics/assay-dev/users/Goutham/4iProcessing-/snakemake_version_testing_output/alignment_parameters")
parser.add_argument("--input_yaml_file", type=str, default="/allen/aics/assay-dev/users/Goutham/4iProcessing-/multiplex_immuno_processing/new_test_outputs/yml_configs/3500005820_4i_modified.yaml")
parser.add_argument("--parent_output_dir", type=str, default="/allen/aics/assay-dev/users/Goutham/4iProcessing-/snakemake_version_testing_output")
parser.add_argument("--round_crop_tempelate", type=str, default="Timelapse")


def max_project(seg_img_labeled):
    xy_seg_maxproj = np.max(seg_img_labeled, axis=0)[np.newaxis, ...][0,:,:]
    return xy_seg_maxproj


def get_align_matrix(alignment_offset):
    align_matrix = np.eye(3)
    for i in range(len(alignment_offset)):
        align_matrix[i, 2] = alignment_offset[i] * -1
    align_matrix = np.int16(align_matrix)
    return align_matrix


def get_shift_to_center_matrix(img_shape, output_shape):
    # output_shape > img_shape should be true for all dimensions
    # and the difference divided by two needs to be a whole integer value

    shape_diff = np.asarray(output_shape) - np.asarray(img_shape)


    shift = shape_diff / 2
    print(f"shift is {shift}")
    
    shift_matrix = np.eye(3)
    for i in range(len(shift)):
        print(f"i is {i}")
        shift_matrix[i, 2] = shift[i]
    shift_matrix = np.int16(shift_matrix)
    return shift_matrix


def load_zstack_mip(filepath, refrence_channel, scene):
    reader = AICSImage(filepath)
    reader.set_scene(int(scene-1)) # b/c of zero indexing ---- this is not reflected in ZEN GUI
    img = reader.data[-1, refrence_channel, :, :, :] # getting T, ch, Z, Y, X
    
    return max_project(img)


def get_FOV_shape(filepath):
    reader = AICSImage(filepath)

    return np.shape(reader)


class perform_alignment_per_position():
    def __init__(self, alignment_csv_dir, yaml_config, parent_output_dir, round_to_crop_to):
        df = pd.read_csv(alignment_csv_dir)
        cond = df["rounds"]==round_to_crop_to
        matching_indices = df[cond].index
        matching_rows = df.loc[matching_indices]
        remaining_rows = df.loc[~cond]
        self.position_csv = pd.concat([matching_rows, remaining_rows]) # makes sure 


        with open(yaml_config) as f:
            self.yaml_config = yaml.load(f, Loader=SafeLoader)


        self.round_to_crop_tempelate = round_to_crop_to
        self.position = os.path.basename(alignment_csv_dir)
        assert os.path.exists(parent_output_dir), "parent output dir doesn't exist"

        self.save_aligned_images_dir = os.path.join(parent_output_dir, "aligned_images")
        if not os.path.exists(self.save_aligned_images_dir):
            os.mkdir(self.save_aligned_images_dir)
        
    def get_tempelate_ref(self, y_dim, x_dim):
        tempelate_ref = np.uint16(np.asarray([y_dim+ (y_dim * 0.33), x_dim + (x_dim * 0.33),]))
        return tempelate_ref
    

    def registeration_using_alignment_params(self, raw_mip, tempelate_ref, alignment_offset):
        """Construct homography matrix and do alignment"""
        shift_to_center_matrix = get_shift_to_center_matrix(raw_mip.shape, tempelate_ref)
        align_matrix = get_align_matrix(alignment_offset)
        combo = shift_to_center_matrix @ align_matrix # matrix multiplication
        aligned_mip = affine_transform(raw_mip, np.linalg.inv(combo), output_shape=tempelate_ref, order=0)
        return aligned_mip


    def find_padding_dimensions(self, padded_img):
        nonzero_rows, nonzero_cols = np.nonzero(padded_img)
        min_y, max_y = min(nonzero_rows), max(nonzero_rows)
        min_x, max_x = min(nonzero_cols), max(nonzero_cols)
        return (min_y, max_y, min_x, max_x)

    def crop_according_to_refrence_crop_round(self, aligned_mip, crop_dims):

        cropped_img = aligned_mip[crop_dims[0]: crop_dims[1], crop_dims[2]: crop_dims[3]]

        return cropped_img

    def save_mip(self, image, round, position, channel):
        # save the mip
        barcode = self.yaml_config["barcode"]
        name = f"{barcode}_R{round}_P{position}_mip_C{channel}.tiff"
        tifffile.imwrite(os.path.join(self.save_aligned_images_dir,name), image)




    def perform_alignment(self):
        # do the alignment
        # refrence_round_info = self.matched_position_csv.loc[self.matched_position_csv['REFRENCE_ROUND']==True].iloc[0]
        # ref_mip = max_project(load_zstack_to_align(refrence_round_info["RAW_filepath"], 2, refrence_round_info["Scene"]).astype(np.uint16))
        
        # alignment_offset = ast.literal_eval(refrence_round_info["cross_cor_params"])
        # alignment_offset = [alignment_offset[1], alignment_offset[2]]
        # print(f"alginment offset is {alignment_offset}")
        # #tempelate_ref = np.zeros((np.shape(ref_zstack)[0]* 2, np.shape(ref_zstack)[1] + (np.shape(ref_zstack)[1] * 0.33), np.shape(ref_zstack)[2] * 2)).astype(np.uint16)



        # #tempelate_ref = np.uint16(np.asarray([np.shape(ref_zstack)[0]* 2, np.shape(ref_zstack)[1] + (np.shape(ref_zstack)[1] * 0.33),np.shape(ref_zstack)[2] + (np.shape(ref_zstack)[2] * 0.33),]))
        # tempelate_ref = np.uint16(np.asarray([np.shape(ref_mip)[0] + (np.shape(ref_mip)[0] * 0.33),np.shape(ref_mip)[1] + (np.shape(ref_mip)[1] * 0.33),]))


        # shift_to_center_matrix = get_shift_to_center_matrix(ref_mip.shape, tempelate_ref)

        # align_matrix = get_align_matrix(alignment_offset)
        # print(f"align matrix is {align_matrix}")
        # print(f"shift center matrix is {shift_to_center_matrix}")
        
        
        # combo = shift_to_center_matrix @ align_matrix # matrix multiplication
        # print(f"combo is {combo}")
        # # aligned image

        # aligned_mip = affine_transform(ref_mip, np.linalg.inv(combo), output_shape=tempelate_ref, order=0)

        # # cropped_maxz = np.max(
        # #     processed_volume, axis=0
        # # )  # compute max z projection
        # import pdb
        # pdb.set_trace()
        

        # for each row in csv
        # Round to crop to first
        # do alignment for each channel
        # then crop according to round to crop too
        # organize by that round first

        _, _,_, y_dim_ref, x_dim_ref = get_FOV_shape(self.position_csv.iloc[0]["RAW_filepath"])
        tempelate_ref = self.get_tempelate_ref(y_dim_ref, x_dim_ref)



        # refrence_round_to_crop = self.matched_position_csv.loc[self.matched_position_csv['Round']==self.round_to_crop_tempelate].iloc[0]
        # shape_tempelate_ref = np.shape(max_project(self.load_zstack_to_align(refrence_round_info["RAW_filepath"], 2, refrence_round_info["Scene"])))
        aligned_positions = []
        for i in range(np.shape(self.position_csv)[0]):
            #for channel in np.shape()
            round_of_intrest = self.position_csv.iloc[i]
            t_dim, ch_dim, z_dim, y_dim, x_dim = get_FOV_shape(round_of_intrest["RAW_filepath"])
            #TODO: align all timepoints
            for ch in range(ch_dim):
                mip_to_align = load_zstack_mip(round_of_intrest["RAW_filepath"], ch, round_of_intrest["Scene"])
                params_to_align_all = ast.literal_eval(round_of_intrest["cross_cor_params"])
                params_to_align_yx = [params_to_align_all[1], params_to_align_all[2]]
                aligned_mip = self.registeration_using_alignment_params(mip_to_align, tempelate_ref, params_to_align_yx)
                if i==0 and ch==0:
                    crop_dims = self.find_padding_dimensions(aligned_mip)

                final_mip = self.crop_according_to_refrence_crop_round(aligned_mip, crop_dims)
                self.save_mip(final_mip, round_of_intrest["rounds"], round_of_intrest["positions"], ch)




if __name__ == "__main__":

    args=parser.parse_args()

    filenames = [f for f in os.listdir(args.input_matched_position_csv_dir) if f.endswith(".csv") and not f.startswith(".")]

    for filename in filenames:
        print(os.path.join(args.input_matched_position_csv_dir, filename))
        position_aligner = perform_alignment_per_position(os.path.join(args.input_matched_position_csv_dir, filename), args.input_yaml_file, args.parent_output_dir, args.round_crop_tempelate)
        position_aligner.perform_alignment()












