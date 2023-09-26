import argparse
from collections import namedtuple
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import ast
from aicsimageio import AICSImage
import pdb
import xml.etree.ElementTree as ET
import lxml.etree as etree
import re
from core.utils import get_round_info_from_dict, get_available_positions, find_matching_position_scene


parser = argparse.ArgumentParser()
parser.add_argument("--input_yaml", type=str, required=True, help="yaml config path")
parser.add_argument("--refrence_round", type=str, default="R1", required=False, help="refrence round to algin to")

# For each postiion
# have a dataframe with columns: Round, scene, align_channel, Refrence_round(True/False)

class create_registration_matching_dataset():
    def __init__(self, yaml_file, refrence_round):

        with open(yaml_file) as f:
            self.yaml_config = yaml.load(f, Loader=SafeLoader)

        self.refrence_round = refrence_round
        
        assert os.path.exists(self.yaml_config["output_path"]), "output path doesn't exist"

        if not os.path.exists(os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"]))):
            os.mkdir(os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"])))

        self.output_matched_csvs_dir = os.path.join(self.yaml_config["output_path"], str(self.yaml_config["barcode"]), "matched_datasets")

        if not os.path.exists(self.output_matched_csvs_dir):
            os.mkdir(self.output_matched_csvs_dir)

    def eliminate_positions_imaged_multiple_times(self, dat):
        """dat(pd.DataFrame) gets rid of rows that have the same round. Assumes that the round name that you want to keep has an "_" in it
        e.g. say there is a row for R2 and R2_B6 present. This function keeps R2_B6 and gets rid of R2 for this position 
        """
        data = dat.copy()
        round_infos = list(data['Round'])
        multiple_matches = [f for f in round_infos if "_" in f]

        for i in range(len(multiple_matches)):
            data = data.loc[data['Round']!=multiple_matches[i].split("_")[0]]
        return data

    def create_dataset_for_position(self, data, ref_round_info, ref_pos, ref_scene, ref_well_id, eliminate_positions_imaged_multiple_times_in_same_round=True):
        """creates a csv for a specific position"""
        all_rounds_list = [data['Data'][f]['round'] for f in range(len(data['Data']))]
        all_rounds_list.remove(self.refrence_round)

        ROUNDS=[]
        POSITIONS = []
        SCENES=[]
        REF_CHANNELS = []
        REFRENCE_ROUND=[]
        RAW_FILEPATH = []
        Well_ids = []
        ROUNDS.append(self.refrence_round)
        POSITIONS.append(ref_pos)
        SCENES.append(ref_scene)
        REFRENCE_ROUND.append(True)
        REF_CHANNELS.append(ref_round_info['ref_channel'])
        RAW_FILEPATH.append(ref_round_info['path'])
        Well_ids.append(ref_well_id)
        print(f"ref pos is {ref_pos}")

        for round in all_rounds_list:
            print(f"looking at round {round}")
            round_info_for_round_to_align = get_round_info_from_dict(round, data)

            round_positions, round_scenes, well_ids_all = get_available_positions(round_info_for_round_to_align)
            try:
                position_to_align, scene_to_align, corresponding_well_id = find_matching_position_scene(round_positions, round_scenes, well_ids_all, ref_pos) # This is the corresponding position/scene combination to align to the refrence round
            except:
                # this is here for cases where the round itself does not have a match
                continue

            ROUNDS.append(round)
            POSITIONS.append(position_to_align)
            Well_ids.append(corresponding_well_id)
            SCENES.append(scene_to_align)
            REF_CHANNELS.append(round_info_for_round_to_align['ref_channel'])
            REFRENCE_ROUND.append(False)
            RAW_FILEPATH.append(round_info_for_round_to_align['path'])

        Position_matched_dataset = pd.DataFrame({'Round': ROUNDS, 'Position': POSITIONS, 'Scene': SCENES, 'Reference_Channel': REF_CHANNELS, 'Well_id': Well_ids, 'REFRENCE_ROUND': REFRENCE_ROUND, 'RAW_filepath': RAW_FILEPATH})
        

        # Position_matched_dataset.drop_duplicates(subset=["rounds"], keep="last", inplace=True)
        if eliminate_positions_imaged_multiple_times_in_same_round==True:
            Position_matched_dataset = self.eliminate_positions_imaged_multiple_times(Position_matched_dataset)

        return Position_matched_dataset
    
    def save_matched_dataset(self, Position_matched_dataset, ref_pos):
        Position_matched_dataset.to_csv(os.path.join(self.output_matched_csvs_dir, f'Position_{str(ref_pos).zfill(2)}.csv'), index=False)

    def create_dataset(self):
        """Gets the positions in the refrence round, and for each position, finds the matches in all other rounds"""
        ref_round_info = get_round_info_from_dict(self.refrence_round, self.yaml_config)
        ref_positions, ref_scenes, ref_well_ids = get_available_positions(ref_round_info)
        print(f"refrence posiiton list is {ref_positions}")

        for ref_pos, ref_scene, ref_well_id in zip(ref_positions, ref_scenes, ref_well_ids):
            print(ref_pos)
            Position_matched_dataset = self.create_dataset_for_position(self.yaml_config, ref_round_info, ref_pos, ref_scene, ref_well_id)
            self.save_matched_dataset(Position_matched_dataset, ref_pos)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = create_registration_matching_dataset(args.input_yaml, args.refrence_round)
    dataset.create_dataset()














    

