from fileinput import filename
import numpy as np
import os
from PIL import Image
import pandas as pd
import argparse
import shutil
from yaml.loader import SafeLoader
import yaml
import ast
from glob import glob
# This script will create a consolidated file with a "best alignment" column that will contain the best parameters to use

# Context: Derek discovered that some of the positions were poorly aligned b/w the timelpase and round 1..... but the other rounds in the same position were perfect. After trying the ORB method, we found that this method does a significantly better job at aligning b/w these two rounds
# This is because the nuclei dye in the timelapse does not penetrate the colony entirely whereas in round 1, the reference nuclei channel tags all the nuclei uniformly. Therefore, the cross correlation understandably stuggles when trying to align very different looking images. The ORB method, which is based on matching invarient features across both images, is better able to align these images

parser = argparse.ArgumentParser()

parser.add_argument("--barcode", type=str, default="5500000724")
parser.add_argument("--output_path", type=str, default="/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports")       
parser.add_argument("--output_dir_merged", default="/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/alignment_csvs_v3_merged", type=str, help="output directory of where to save files")
parser.add_argument("--position_list_to_use_ORB", nargs="*", default=[], type=str, help="position list")


def get_basename(name):
    return os.path.basename(name)




def generate_merged_manifest(input_path, best_method="ORB"):
    # function to process dataframe and create new dataframe based on method chosen
    # IF ORB---> use ORB method for timelapse and cross corr for everything else in best_alignment_params
    # IF cross_corr ---> use cross corr for everything

    if best_method == "ORB":
        df = pd.read_csv(input_path)
        df["best_alignment_params"] = np.nan
        best_alignment_params = []
        for i in range(np.shape(df)[0]):
            if df.loc[i]["key"] == "20X_Timelapse":
                best_alignment_params.append(ast.literal_eval(df.loc[i]["alignment_offsets_zyx"]))
            else:
                best_alignment_params.append(ast.literal_eval(df.loc[i]["alignment_offsets_zyx_cross_cor"]))
        df["best_alignment_params"] = best_alignment_params
    else:
        df = pd.read_csv(input_path)
        df["best_alignment_params"] = np.nan
        best_alignment_params = []
        for i in range(np.shape(df)[0]):
            best_alignment_params.append(ast.literal_eval(df.loc[i]["alignment_offsets_zyx"]))
        df["best_alignment_params"] = best_alignment_params

    ### 

    return df







if __name__ == "__main__":
    args = parser.parse_args()

    yaml_dir = os.path.join(args.output_path, "yml_configs")

    yaml_list = [x for x in os.listdir(yaml_dir) if "_confirmed" in x]
    yaml_list = [x for x in yaml_list if args.barcode in x]
    dflist = []
    for y in yaml_list:
        print(y)
        yml_path = yaml_dir + os.sep + y
        with open(yml_path) as f:
            data = yaml.load(f, Loader=SafeLoader)
            for round_dict in data["Data"]:

                dfsub = pd.DataFrame(round_dict.values(), index=round_dict.keys()).T
                dfsub["barcode"] = data["barcode"]
                dfsub["scope"] = data["scope"]
                dfsub["output_path"] = data["output_path"]
                dflist.append(dfsub)

    dfconfig = pd.concat(dflist)
    dfconfig.set_index(["barcode", "round"], inplace=True)


    output_dir = dfconfig["output_path"][0]
    align_csv_dir_crosscor = output_dir + os.sep + "alignment_csvs_each_" + "cross_cor"
    align_csv_dir_ORB = output_dir + os.sep + "alignment_csvs_each_" + "ORB"
    align_csv_name_glob = f"{args.barcode}*alignment_csv_each.csv"
    print(align_csv_dir_crosscor)
    print(align_csv_dir_ORB)
    globlist_cross_cor = list(map(get_basename, glob(align_csv_dir_crosscor + os.sep + align_csv_name_glob)))
    globlist_ORB = list(map(get_basename, glob(align_csv_dir_ORB + os.sep + align_csv_name_glob)))
    print("cross cor is {}".format(globlist_cross_cor))
    print("ORB is {}".format(globlist_ORB))
    for pos in args.position_list_to_use_ORB:
        #print(f"pos is {pos}")
        #print(f"pos is type {type(pos)}")
        print(f"pos is {pos}")
        position = pos + "-"
        filename_ORB = [f for f in globlist_ORB if position in f]
        #print(filename_ORB)
        #print("found")
        
        assert len(filename_ORB)==1, f"mismatch in number of files. total is {len(filename_ORB)} and matching with {filename_ORB}"
        df_final = generate_merged_manifest(os.path.join(align_csv_dir_ORB, filename_ORB[0]))
        #print(len(globlist_cross_cor))
        print(f"removing {filename_ORB[0]}")
        globlist_cross_cor.remove(filename_ORB[0])

        df_final.to_csv(os.path.join(args.output_dir_merged,filename_ORB[0]))
        #print(len(globlist_cross_cor))



    #print(globlist)
    #filenames_ORB = [f for f in os.listdir(args.align_csv_dir_crosscor)]
    dfalign_list = []
    # Indexing position
    for align_csv_path in globlist_cross_cor: #If present here use ORB for round 1 + cross corr for others
        # align_csv_path = align_csv_dir + os.sep + align_csv_name
        #df = pd.read_csv(os.path.join(align_csv_dir_crosscor,align_csv_path)) #

        df_final = generate_merged_manifest(os.path.join(align_csv_dir_crosscor, align_csv_path), best_method="cross_cor")

        print(f"align_csv_path is {os.path.basename(align_csv_path)}") 
        cols_to_delete= [f for f in list(df_final.columns) if f.startswith("Unnamed")]
        df_final=df_final.drop(columns = cols_to_delete, axis = 1)
        df_final.to_csv(os.path.join(args.output_dir_merged, align_csv_path))


    # If not present here, get info from ORB


        #dfalign_list.append(df)
    #dfalign = pd.concat(dfalign_list)