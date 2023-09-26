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









class czi_metadata_helper():
    # TODO: modify with othere get functions that return other useful metadata... currently set up to only return position/scene number but can change that in the future
    def __init__(self, czi_filepath):
        self.img = AICSImage(czi_filepath)
        metadata_raw = self.img.metadata
        self.metastr = ET.tostring(metadata_raw).decode("utf-8")
        self.metadata = etree.fromstring(self.metastr)

    def get_well_id_from_scene(self, scene):
        """From xml element, extract well id"""
        shape = scene.find(".//Shape")
        well_id = shape.get("Name")
        # zero pad the well_id number by two
        non_numeric_part = ''.join(filter(lambda x: not x.isdigit(), well_id))
        numeric_part = ''.join(filter(lambda x: x.isdigit(), well_id))
        well_id_str = non_numeric_part + f"{str(numeric_part).zfill(2)}"
        return well_id_str

    def debug_save_metadata_as_xml(self):
        """For debuggging purposes, we save the metadata as xml"""
        with open("output_xml_debug.xml", "w") as xml_file: 
            xml_file.write(self.metastr)
        
        
    def get_position_scene_wellid_paired_list(self):
        scenes = self.metadata.findall(".//Scenes/Scene")
        print(len(scenes))
        scenes_list = []
        positions_list = []
        well_ids_list = []

        for scene in scenes:
            scene_index = scene.get("Index")
            scene_name = scene.get("Name")
            center_position = scene.find(".//CenterPosition").text
            well_id = self.get_well_id_from_scene(scene)
            scenes_list.append(int(scene_index)+1)
            positions_list.append(int(re.findall(r'\d+', scene_name)[0]))
            well_ids_list.append(well_id)

        return positions_list, scenes_list, well_ids_list
    