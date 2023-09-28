import argparse
import os
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import zen_position_helper

#pd.set_option("display.max_columns", None)
#ploton = False


"""
This script loads the yaml config files and populate a dataframe with config info
1. find the yaml files for each barcode or a single barcode
2. retrieve positions/scenes/coordinates from experiment file for acquired positions
"""


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)
parser.add_argument(
    "--barcode",
    type=str,
    required=False,
    help="optional arg to only run a single barcode if desired"
)

if __name__ == "__main__":
    args = parser.parse_args()

    yaml_dir = os.path.join(args.output_path, "yml_configs")
    yaml_list = [x for x in os.listdir(yaml_dir) if "_confirmed" in x]
    dfconfiglist = []

    if args.barcode:
        yaml_list = [x for x in yaml_list if x.startswith(args.barcode)]
        assert len(yaml_list) ==1, f"mismatch in files found"
    


    for y in yaml_list:
        print(y)
        yml_path = yaml_dir + os.sep + y
        with open(yml_path) as f:
            print(f"yaml path is {f}")
            data = yaml.load(f, Loader=SafeLoader)
            for round_dict in data["Data"]:
                dfconfigsub = pd.DataFrame(
                    round_dict.values(), index=round_dict.keys()
                ).T
                dfconfigsub["barcode"] = data["barcode"]
                dfconfigsub["scope"] = data["scope"]
                dfconfigsub["output_path"] = data["output_path"]
                dfconfiglist.append(dfconfigsub)

    dfconfig = pd.concat(dfconfiglist)
    dfconfig.set_index(["barcode", "round"], inplace=True)

    # now go through all specified CZI files and collect metadata
    # (specifically metadata about position, XYZ coordinates and FOV size)
    # barcode corresponds to a given plate
    # round corresponds to a given round of imaging of that plate
    for barcode, dfcb in dfconfig.groupby(["barcode"]):
        dfmeta_barcode_list = []
        for round, dfcbr in dfcb.groupby(["round"]):
             
            print(barcode, round)

            # determine if flist is list of filepaths or fms fileids
            # and if it is fms ID then find the filepath
            file_list = []
            original_file_list = []
            list_of_files = dfcbr.path.tolist()
            # import pdb
            # pdb.set_trace()
            for file in list_of_files:
                original_file_list.append(file)
                if os.path.exists(file):
                    file_list.append(file)
                else:  # try to find FMS id - currently not set up for fms
                    # file = fms.get_file_by_id(file)
                    # file_list.append('/'+file.path)
                    # print(file,'-->',file.path)
                    assert os.path.exists(file), f"czi file not found, update your yaml... cant find {file}"
            # get position info from all files in the file list
            dfmeta_round_list = []
            if len(file_list) > 0:
                # each round may have more than one czi file assoicated with it. so iterate through each file
                for original_file, filename in zip(original_file_list, file_list):
                    print(file, filename)
                    dfmeta_sub = zen_position_helper.get_position_info_from_czi(
                        filename
                    )
                    dfmeta_sub["align_channel"] = dfcbr["ref_channel"][0]
                    dfmeta_sub["barcode"] = barcode
                    dfmeta_sub["key"] = round
                    dfmeta_sub[
                        "original_file"
                    ] = original_file  # this records the original item from the yaml
                    dfmeta_round_list.append(dfmeta_sub)

                dfmeta_round = pd.concat(
                    dfmeta_round_list
                )  # this has all metadata for given round

                dfmeta_barcode_list.append(dfmeta_round)

        dfmeta = pd.concat(
            dfmeta_barcode_list
        )  # this has all metadata for all rounds of image data for one given barcode

        # important columns are:
        # ['original_file',
        #  'file',
        #  'parent_file',
        #  'imgsize_pixels',
        #  'ImagePixelDistances',
        #  'totalmagnification',
        #  'channel_dict',
        #  'pixelSizes',
        #  'imgsize_um',
        #  'PlateAnchorPoint',
        #  'PlateReferencePoint',
        #  'X',
        #  'Y',
        #  'Z',
        #  'IsUsedForAcquisition',
        #  'Position',
        #  'Position_num',
        #  'Scene',
        #  'Well_id',
        #  'X_original',
        #  'X_adjusted',
        #  'Y_original',
        #  'Y_adjusted',
        #  'align_channel',
        #  'barcode',]

        output_dir = dfconfig["output_path"][0]
        csv_dir = output_dir + os.sep + "csvs"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_name = barcode + "metadata_csv.csv"
        csv_path = csv_dir + os.sep + csv_name
        print("\n\n" + csv_path + "\n\n")
        dfmeta.to_csv(os.path.abspath(csv_path))

        

        # this is the key information (along with the info in df config)
        # df[['barcode','key','Position','Scene','fname','AcquisitionTime']]
