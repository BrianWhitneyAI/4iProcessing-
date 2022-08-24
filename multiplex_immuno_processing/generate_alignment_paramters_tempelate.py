import argparse
import os
import skimage.exposure as skex
from aicsimageio import AICSImage
import matplotlib as plt
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import registration_utils
import jinja2
import subprocess

overwrite = True

# this code computes the alignment parameters to align each scene across the multiple rounds of imaging
# should take a barcode as an argument,
# then this code reads that dataframe pickle for that given barcode
# then it uses the dataframe to determine the reference channel to be used for each of the rounds for alignment
# then it loads all the reference channel images for each round
# then it runs the an alignment algorithm to align all of the rounds
# all positions should be aligned to round 1 (that will be the reference round)
# this will enable all the processing to be run in stages later on as new data is acquire

# print(sys.argv)
# arg_list = [
#     (x.replace("--", ""), i)
#     for i, x in enumerate(list(sys.argv))
#     if bool(re.search("--", x))
# ]
# args_dict = {}
# for keyval in arg_list:
#     args_dict[keyval[0]] = sys.argv[keyval[1] + 1]
# print(args_dict)
# print()
# print(args_dict["barcode"])
# print()

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



if __name__ == "__main__":
    args = parser.parse_args()
    cwd = os.getcwd()

    template_dir = f'{cwd}'


    # Open the file and load the file
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
    pickle_dir = output_dir + os.sep + "pickles"
    pickle_name = barcode + "cleanedup_match_pickle.pickle"
    pickle_path = pickle_dir + os.sep + pickle_name
    print("\n\n" + pickle_path + "\n\n")
    print(os.path.exists(pickle_path))
    # dfall = pd.read_pickle(pickle_path)

    pickle_name = barcode + "cleanedup_match_csv.csv"
    pickle_path = pickle_dir + os.sep + pickle_name
    print("\n\n" + pickle_path + "\n\n")
    print(os.path.exists(pickle_path))
    dfall = pd.read_csv(pickle_path)

    dfall["parent_file"] = dfall["parent_file"].apply(lambda x: os_swap(x))

    template_position_list = dfall["template_position"].unique()
    # keylist = mag_dict.keys()
    keylist = dfall["key"].unique()
    # for Position in ['P2']:

    print(template_position_list)
    dfkeeplist = []

    j2env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir)
    )
    # variables used for each position
    # Position
    # dfall
    # 
    output_csv_path = os.path.join(args.output_path, "position_csvs_generate_align_params")


    if not os.path.exists(output_csv_path):
        os.mkdir(output_csv_path)
    dfall.to_csv(os.path.join(output_csv_path, f"barcode_{barcode}_all.csv"))

    for i in range(len(template_position_list)):
        
        position = template_position_list[i]
        render_dict_slurm = {
        'dfall_csv_dir': os.path.join(output_csv_path, f"barcode_{barcode}_all.csv"),
        'barcode': barcode,
        'output_path': args.output_path,
        'position': position,
        }
        print(render_dict_slurm)
        template_slurm = j2env.get_template('run_generate_alignment_params.j2')
        this_script = template_slurm.render(render_dict_slurm)
        script_path = os.path.join("/allen/aics/assay-dev/users/Goutham/4iProcessing-/multiplex_immuno_processing/jinja_out", f"barcode_{barcode}_position_{template_position_list[i]}_alignment_param.script")  # noqa E501
        with open(script_path, 'w') as f:
            f.writelines(this_script)        
        submission = "sbatch " + script_path
        print("Submitting command: {}".format(submission))
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)  # noqa E501
        (out, err) = process.communicate()



