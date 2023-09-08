import os
import argparse
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from aicsimageio import AICSImage
import numpy as np
import registration_utils
import tifffile

"""
This step reads in the matched position csv and finds the alignment displacement for each round to the refrence round
Next, it aligns maxprojects of the images and saves out the registered images and updates the matched position csvs
with the corresponding displacements for each round. This information is needed is saved out so we can optionally align labelfree predictions if needed
"""


def max_project(seg_img_labeled):
    xy_seg_maxproj = np.max(seg_img_labeled, axis=0)[np.newaxis, ...][0,:,:]
    return xy_seg_maxproj


parser = argparse.ArgumentParser()
parser.add_argument("--input_yaml", type=str, required=True, help="yaml config path")
parser.add_argument("--matched_position_csv_dir", type=str, required=False, help="Matched position csv to align a single position (Optional)")



class Position_aligner():
    """Aligns positions by taking each corresponding csv and finding alignment parameters and performing the registration(different function- in registration utils)"""
    def __init__(self, matched_position_csv_dir, yaml_config):
        self.matched_position_csv = pd.read_csv(matched_position_csv_dir)
        self.position = os.path.basename(matched_position_csv_dir)
        
        with open(yaml_config) as f:
            self.yaml_config = yaml.load(f, Loader=SafeLoader)

        assert os.path.exists(self.yaml_config["output_path"]), "parent output dir doesn't exist"
        
        assert os.path.exists(os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"]))), "parent dir doesn't exist"
        self.save_aligned_csv_dir = os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"]), "alignment_parameters")
        if not os.path.exists(self.save_aligned_csv_dir):
            os.mkdir(self.save_aligned_csv_dir)

    def load_zstack_to_align(self, filepath, refrence_channel, scene):
        reader = AICSImage(filepath)
        reader.set_scene(int(scene-1)) # b/c of zero indexing ---- this is not reflected in ZEN GUI
        try:
            align_channel_index = [xi for xi, x in enumerate(reader.channel_names) if x == refrence_channel][0]        
        except:
            align_channel_index = refrence_channel
        img = reader.data[-1, align_channel_index, :, :, :] # getting T, ch, Z, Y, X
        
        return img

    def save_csv_alignment(self, dat):
        dat.to_csv(os.path.join(self.save_aligned_csv_dir, self.position), index=False)



    def create_aligned_dataset(self):
        # Pseudocode
        # load csv, get each round information for that position and the refrence channel
        refrence_round_info = self.matched_position_csv.loc[self.matched_position_csv['REFRENCE_ROUND']==True].iloc[0]
        rounds_to_align = self.matched_position_csv.loc[self.matched_position_csv['REFRENCE_ROUND']==False]
        ref_zstack = self.load_zstack_to_align(refrence_round_info["RAW_filepath"], 3, refrence_round_info["Scene"])

        print(np.shape(ref_zstack))
        alignment_parameters_cross_corr = []
        for i in range(np.shape(self.matched_position_csv)[0]):
            round_info = self.matched_position_csv.iloc[i]

            if round_info["rounds"] == refrence_round_info["rounds"]:
                meanoffset = [0, 0, 0]
                alignment_parameters_cross_corr.append(meanoffset)
                continue
            if round_info["rounds"] == "Timelapse":
                to_align_zstack = self.load_zstack_to_align(round_info["RAW_filepath"], 2, round_info["Scene"])
            else:
                to_align_zstack = self.load_zstack_to_align(round_info["RAW_filepath"], 3, round_info["Scene"])

            
            (_, _, meanoffset, _,) = registration_utils.find_zyx_offset(
                ref_zstack.copy(), to_align_zstack.copy(), ploton=False, verbose=False,
            )
            alignment_parameters_cross_corr.append(meanoffset)

        self.matched_position_csv["cross_cor_params"] = alignment_parameters_cross_corr

        self.save_csv_alignment(self.matched_position_csv)


if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(args.input_yaml) as f:
        yaml_config = yaml.load(f, Loader=SafeLoader)


    if args.matched_position_csv_dir:
        registration_dataset = Position_aligner(args.matched_position_csv_dir, args.input_yaml)
        registration_dataset.create_aligned_dataset()

    else:
        input_matched_position_csv_dir = os.path.join(yaml_config["output_path"], str(yaml_config["barcode"]), "matched_datasets")
        assert os.path.exists(input_matched_position_csv_dir), "input matched position dir doesn't exist"
        filenames = [f for f in os.listdir(input_matched_position_csv_dir) if f.endswith(".csv") and not f.startswith(".")]

        for file in filenames:
            registration_dataset = Position_aligner(os.path.join(input_matched_position_csv_dir, file), args.input_yaml)
            registration_dataset.create_aligned_dataset()




