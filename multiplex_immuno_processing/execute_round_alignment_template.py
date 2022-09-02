import argparse
from glob import glob
import os
from pathlib import Path
import re
import registration_utils
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
import skimage.io as skio
import yaml
from yaml.loader import SafeLoader
import jinja2
import subprocess

def os_swap(x):
    out = "/" + ("/".join(x.split("\\"))).replace("//", "/")
    return out

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)
parser.add_argument(
    "--barcode", type=str, required=True, help="specify barcode to analyze"
)
parser.add_argument(
    "--method", choices=['cross_cor', 'ORB'])

parser.add_argument("--position_list", nargs="+", default=[], type=list, help="input dirs to parse")


if __name__ == "__main__":
    args = parser.parse_args()

    # Open the file and load the file
    # Open the file and load the file
    cwd = os.getcwd()
    template_dir = f'{cwd}'
    j2env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir)
    )

    barcode = args.barcode

    yaml_dir = os.path.join(args.output_path, "yml_configs")

    yaml_list = [x for x in os.listdir(yaml_dir) if "_confirmed" in x]
    yaml_list = [x for x in yaml_list if barcode in x]
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

    mag = "20x"

    output_dir = dfconfig["output_path"][0]
    csv_dir = output_dir + os.sep + "csvs"
    csv_name = barcode + "cleanedup_match_csv.csv"
    csv_path = csv_dir + os.sep + csv_name
    print("\n\n" + csv_path + "\n\n")
    # dfall = pd.read_csv(csv_path)
    dfall = pd.read_csv(csv_path)


    output_dir = dfconfig["output_path"][0]
    align_csv_dir = output_dir + os.sep + "alignment_csvs_each_" + args.method
    align_csv_name_glob = f"{barcode}*alignment_csv_each.csv"
    print(align_csv_name_glob)
    globlist = glob(align_csv_dir + os.sep + align_csv_name_glob)
    dfalign_list = []
    for align_csv_path in globlist:
        # align_csv_path = align_csv_dir + os.sep + align_csv_name
        df = pd.read_csv(align_csv_path)
        dfalign_list.append(df)
    dfalign = pd.concat(dfalign_list)

    dfall["parent_file"] = dfall["parent_file"].apply(lambda x: os_swap(x))

    # merge both dataframes so that you only try to align the positions that can be aligned.
    dfall = pd.merge(
        dfalign,
        dfall,
        on=["key", "template_position"],
        suffixes=("_align", ""),
        how="left",
    )

    keylist = dfall["key"].unique()
    # for Position in ['P2']:

    output_csv_path = os.path.join(args.output_path, "alignment_csvs_each_" + args.method)


    if not os.path.exists(output_csv_path):
        os.mkdir(output_csv_path)
    dfall.to_csv(os.path.join(output_csv_path, f"barcode_{barcode}_all.csv"))

    if not args.position_list:
        template_position_list = dfall["template_position"].unique()
    else:
        template_position_list = []
        for i in range(len(args.position_list)):
            template_position_list.append(''.join(args.position_list[i]))

    print(template_position_list)

    for i in range(len(template_position_list)):
        position = template_position_list[i]
        render_dict_slurm = {
        'dfall_csv_dir': os.path.join(output_csv_path, f"barcode_{barcode}_all.csv"),
        'barcode': barcode,
        'output_path': args.output_path,
        'position': position,
        'method': args.method,
        'jinja_output': os.path.join(args.output_path, "jinja_output"),
        'cwd': os.getcwd()
        }
        

        print(render_dict_slurm)

        template_slurm = j2env.get_template('run_alignment.j2')
        this_script = template_slurm.render(render_dict_slurm)
        script_path = os.path.join(args.output_path, "jinja_out", f"barcode_{barcode}_position_{template_position_list[i]}.script")  # noqa E501
        with open(script_path, 'w') as f:
            f.writelines(this_script)
        
        submission = "sbatch " + script_path
        print("Submitting command: {}".format(submission))
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)  # noqa E501
        (out, err) = process.communicate()





