import argparse
import os

import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

pd.set_option("display.max_columns", None)
ploton = False

# now you need to resolve the ambiguities
"""
1. overlapping FOVs within the same round
--first check if any are marked as a scene to toss. If so, then remove that those from the list of options.
--if multiple options still exist, then choose the FOV that was imaged first
--(if its from the same parent file, this can be determined by scene number;
--otherwise this has to be determined from aqcuisition time metadata)
-- this needs to be removed from the template position list

2. a position can't be matched back to the template session
--this position can still be processed but it is not useful for 4i
--it won't be linked for analysis so it is ok to keep in the dataframae

3. there are multiple positions that overlap with the same reference position
--first check if any are marked as a scene to toss. If so, then remove that those from the list of options.
--if multiple options still exist then, then check if one image overlaps more than another,
--if two images overlap to similar degrees, then choose the image that was acquired first.
--(if its from the same parent file, this can be determined by scene number;
--otherwise this has to be determined from aqcuisition time metadata)
--TODO: get image time metadata.
--NOTE: multiple overlaps can happen if multiple positions in the same FOV overlap...
--because these will then overlap multiple times with the reference position

"""


# interesting_cols = ['key','template_position','template_position_unique',
# 'move_position_unique','move_position','PositionUnique','Position']
# grouper = ['key','template_position','template_position_unique']


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # load the yaml config files and populate a dataframe with config info
    yaml_dir = os.path.join(args.output_path, "yml_configs")
    yaml_list = [x for x in os.listdir(yaml_dir) if "_confirmed" in x]
    dfconfiglist = []
    for y in yaml_list:
        print(y)
        yml_path = yaml_dir + os.sep + y
        with open(yml_path) as f:
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
    # dfconfig = dfconfig[dfconfig['barcode']=='5500000725']
    dfconfig.set_index(["barcode", "round"], inplace=True)

    # now open the metadata pickle
    # (specifically metadata about position, XYZ coordinates and FOV size)
    # barcode corresponds to a given plate
    # round corresponds to a given round of imaging of that plate
    for barcode, dfcb in dfconfig.groupby(["barcode"]):

        output_dir = dfconfig["output_path"][0]
        metadata_pickle_dir = output_dir + os.sep + "pickles"
        metadata_pickle_name = barcode + "matched_positions_pickle.pickle"
        metadata_pickle_path = metadata_pickle_dir + os.sep + metadata_pickle_name

        print(barcode)
        dfmeta = pd.read_pickle(metadata_pickle_path)

        #         ################################################
        #         # first remove positions flagged for removal (this does not work)
        #         ################################################
        #         positions_to_remove = dfmeta[dfmeta['flag-scene_to_toss']==True]['PositionUnique'].unique()

        #         # first remove all instances of these positions as a move_position
        #         dfmeta.set_index(['PositionUnique'],inplace=True)
        #         dfmeta.drop(index = positions_to_remove,
        #                    inplace=True)
        #         dfmeta.reset_index(inplace=True)

        #         positions_to_remove = dfmeta[dfmeta['flag-scene_to_toss']==True]['move_position_unique'].unique()
        #         # now remove all instances of these positions as a template_position_unique
        #         dfmeta.set_index(['template_position_unique'],inplace=True)
        #         dfmeta.drop(index = positions_to_remove,
        #                    inplace=True)
        #         dfmeta.reset_index(inplace=True)

        ################################################
        # find template_positions (i.e. P2) that have multiple template_position_unique
        # (multiple positions are from multiple czi files...only one can be chosen)
        ################################################
        dfover = dfmeta[dfmeta["flag-overlaps_with_another_position_in_same_round"]]
        conflicting_tpu_combo_list = []
        for (key, tp), dfg in dfover.groupby(["key", "template_position"]):
            tpu_list = dfg.template_position_unique.unique()
            if len(tpu_list) > 1:
                conflicting_tpu_combo_list.append(tpu_list)

        print("number of sets of conflicting self-overlaps", len(tpu_list))
        # now check to see if one is flagged for removal and if one was acquired first
        positions_to_flag_as_imaged_second = []
        for tpu_list in conflicting_tpu_combo_list:

            # find the move_position_unique to get acquisition_time info
            dfslice = (
                dfmeta.groupby(["move_position_unique"]).agg("first").loc[tpu_list]
            )

            # assert that all positions should not be flagged for removal
            log = dfslice["flag-scene_to_toss"]
            assert np.sum(log) < dfslice.shape[0]

            # append the positions marked for tossing
            toss_positions = dfslice[log].index.values
            positions_to_flag_as_imaged_second.extend(toss_positions)

            # remove the flagged positions from consideration
            dfslice2 = dfslice[~log]

            # now determine which scene was imaged first
            dfslice2.sort_values(["AcquisitionTime"], ascending=True, inplace=True)

            later_imaged_positions = dfslice2.index.values[1::]

            # append the subsequent imaged positions to mark for removal (i.e. flag for removal)
            positions_to_flag_as_imaged_second.extend(later_imaged_positions)

        # flag for removal the overlapping positions that were already flagged or imaged second
        print("positions_to_flag_as_imaged_second", positions_to_flag_as_imaged_second)

        # first create a new flag columns
        dfmeta["flag-overlapping_positions_within_same_round_imaged_second"] = [
            False
        ] * dfmeta.shape[0]

        # now flag
        dfmeta.set_index(["template_position_unique"], inplace=True)
        dfmeta.loc[
            positions_to_flag_as_imaged_second,
            "flag-overlapping_positions_within_same_round_imaged_second",
        ] = True

        # dfover = dfmeta[dfmeta['flag-overlaps_with_another_position_in_same_round']]

        # remove these duplicates!
        dfmeta = dfmeta[
            dfmeta["flag-overlapping_positions_within_same_round_imaged_second"]==False
        ]
        dfmeta.reset_index(inplace=True)

        # now check that there are the same number of tps and tpus
        tpnum = dfmeta.groupby("template_position").agg("first").shape[0]
        tpunum = dfmeta.groupby("template_position").agg("first").shape[0]
        print(tpnum, tpunum)

        ################################################

        #####################################################################
        # now check for positions that can't be matched back to the reference position (round1)
        #####################################################################
        # these do not require action at this time
        unmatchdf = dfmeta[dfmeta["template_position"] == "NOMATCH"]
        if unmatchdf.shape[0] < 1:
            print("no unmatched positions")
        #####################################################################

        #####################################################################
        # now look for instances where multiple images are matched to the same reference FOV
        #####################################################################
        print("look for multiple matches")

        # you can find these by looking for template position unique that has multiple entries for a given round
        dfg = dfmeta.groupby(["template_position_unique", "key"]).agg("count")
        ivs = dfg[dfg["file"] > 1].index.values

        # create a slice of the dataframe based on the identified indexes
        dfsub = dfmeta.set_index(["template_position_unique", "key"]).loc[ivs]
        subivs = np.unique(dfsub.index.values)

        # this block is for displaying only
        dfsub[
            [
                "move_position_unique",
                "PositionUnique",
                "Position",
                "Scene",
                "X",
                "Y",
                "Z",
                "flag-overlapping_positions_within_same_round_imaged_second",
                "overlap",
            ]
        ]

        print("number of matches to discern thru", len(subivs))

        dfmeta.reset_index(inplace=True)
        # prepare to iterate through index values identified above
        dfmeta.set_index(["template_position_unique", "key"], inplace=True)
        positions_to_flag_as_imaged_second = []
        # for position in tpulist:
        for iv in subivs:

            # print('iv',iv)
            # find the given template_position and round combo
            dfsub = dfmeta.loc[iv].groupby("move_position_unique").agg("first")
            positions_to_test = dfsub.index.values

            # assert that both positions should not be flagged for removal
            log = (dfsub["flag-scene_to_toss"]) | (
                dfsub["flag-overlapping_positions_within_same_round_imaged_second"]
            )
            assert np.sum(log) < len(positions_to_test)

            # append the positions marked for tossing
            toss_positions = dfsub[log].index.values
            positions_to_flag_as_imaged_second.extend(toss_positions)

            # now determine which scene was imaged first
            dfsub2 = dfsub[~log]
            dfsub2.sort_values(["AcquisitionTime"], ascending=True, inplace=True)

            first_imaged_position = dfsub2.index.values[0]
            later_imaged_positions = dfsub2.index.values[1::]

            # append the subsequent imaged positions
            positions_to_flag_as_imaged_second.extend(later_imaged_positions)

        # now do the flagging
        print(
            "multiple_fovs_matched_to_same_reference_FOV_imaged_second",
            positions_to_flag_as_imaged_second,
        )

        dfmeta.reset_index(inplace=True)

        dfmeta.set_index(["move_position_unique"], inplace=True)
        dfmeta["flag-multiple_fovs_matched_to_same_reference_FOV_imaged_second"] = [
            False
        ] * dfmeta.shape[0]
        dfmeta.loc[
            positions_to_flag_as_imaged_second,
            "flag-multiple_fovs_matched_to_same_reference_FOV_imaged_second",
        ] = True

        dfmeta.reset_index(inplace=True)
        dfmeta.set_index(["template_position_unique", "key"], inplace=True)
        dfmeta.loc[
            ivs,
            [
                "flag-overlaps_with_another_position_in_same_round",
                "overlapping_positions_within_same_round",
                "flag-overlapping_positions_within_same_round_imaged_second",
                "flag-multiple_fovs_matched_to_same_reference_FOV_imaged_second",
            ],
        ]

        # now remove those multiple matches
        dfmeta = dfmeta[
            dfmeta["flag-multiple_fovs_matched_to_same_reference_FOV_imaged_second"]==False]

        dfmeta.reset_index(inplace=True)
        grouper = ["key", "template_position", "move_position_unique"]
        dfmeta.set_index(grouper).loc[dfmeta.groupby(grouper).agg("count")["file"] > 1]
        ################################################

        # define useful columns for sorting so it can display nicely
        def round_numberator(x):
            import re

            if bool(re.search("Time", x, re.IGNORECASE)):
                out = 0
            elif bool(re.search("NOMATCH", x, re.IGNORECASE)):
                out = -1

            else:
                search = re.search("[0-9]+", x, re.IGNORECASE)
                assert search is not None
                out = int(search.group(0))

            return str(out).zfill(2)

        dfmeta["round_number"] = dfmeta["key"].apply(lambda x: round_numberator(x))
        dfmeta["template_position_zfill"] = dfmeta["template_position"].apply(
            lambda x: round_numberator(x)
        )

        # reorder the columns
        columns = dfmeta.columns.tolist()
        interesting_cols = [
            "barcode",
            "scope",
            "key",
            "template_position",
            "move_position",
            "Scene",
            "fname",
            "Well_id",
            "template_position_unique",
            "move_position_unique",
            "PositionUnique",
            "Position",
            "AcquisitionTime",
            "align_channel",
            "overlap",
        ]
        cols = interesting_cols + [x for x in columns if x not in interesting_cols]
        dfmeta = dfmeta[cols]

        # now sort for clear display
        dfmeta.sort_values(["round_number", "template_position_zfill"], inplace=True)

        ################################################
        # now save out the pickle and csv
        ################################################
        output_dir = dfconfig["output_path"][0]
        pickle_dir = output_dir + os.sep + "pickles"
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle_name = barcode + "cleanedup_match_pickle.pickle"
        pickle_path = pickle_dir + os.sep + pickle_name
        print("\n\n" + pickle_path + "\n\n")
        dfmeta.to_pickle(os.path.abspath(pickle_path))

        out_csv_path = pickle_path.replace("_pickle", "_csv").replace(".pickle", ".csv")
        dfmeta.to_csv(os.path.abspath(out_csv_path))

        ################################################

        grouper = ["key", "template_position", "move_position_unique"]
        dftest = dfmeta.set_index(grouper).loc[
            dfmeta.groupby(grouper).agg("count")["file"] < 1
        ]
        print("number of template_positions without matches", dftest.shape[0])

        dftest = dfmeta.set_index(grouper).loc[
            dfmeta.groupby(grouper).agg("count")["file"] > 1
        ]
        print("number of template_positions_with_multiple matches", dftest.shape[0])
