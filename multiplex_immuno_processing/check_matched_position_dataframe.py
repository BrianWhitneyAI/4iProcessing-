import argparse
from glob import glob
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    filedir = args.output_path + os.sep + "csvs"

    globout = glob(filedir + os.sep + "*cleanedup_match_csv*csv")
    print(globout)
    print()
    for filepath in globout:
        print(Path(filepath).name)
        df = pd.read_csv(filepath)
        dfg = df.groupby("template_position").agg("count")

        print("number of positions = ", dfg.shape[0])

        log = df["move_position"].values == df["template_position"].values
        dfgp = df.groupby(["template_position", "key"]).agg("first")
        dfgpr = dfgp.reset_index()
        log = dfgpr["move_position"].values == dfgpr["template_position"].values
        print("number of position name mismatches = ", dfgpr.shape[0] - np.sum(log))
        print(dfgpr[~log][["template_position", "move_position", "key", "barcode"]])

        dfmeta_out = df.copy()
        dfg = dfmeta_out.groupby(["template_position_unique", "key"]).agg("count")
        ivs = dfg[dfg["Position"] > 1].index.values
        dfm = dfmeta_out.set_index(["template_position_unique", "key"])
        flagcols = [x for x in dfm.columns.tolist() if "flag" in x]

        dfm = dfm.sort_index(
            level=dfm.index.names
        )  # this is needed to handle warning that arises in pandas

        print(
            dfm.loc[
                ivs,
                [
                    "overlap",
                    "move_position",
                    "Scene",
                    "overlapping_positions_within_same_round",
                ]
                + flagcols,
            ]
        )
        print()

        # determine number of positions
        dfgp = df.groupby(["Position"]).agg("count")
        number_of_positions = dfgp.shape[0]
        print("number of positions = ", number_of_positions)

        # determine number of rounds (keys)
        dfgk = df.groupby(["key"]).agg("count")
        number_of_rounds = dfgk.shape[0]
        print("number of rounds = ", number_of_rounds)

        # determine number of positions that were imaged more or less than once per round
        dfgp_diffnumber = dfgp[dfgp["file"] != number_of_rounds]
        print(
            "number of positions imaged a different number of times than the number of rounds=",
            dfgp_diffnumber.shape[0],
        )
        print(
            "positions that were imaged more than once per round",
            dfgp[dfgp["file"] > number_of_rounds].index.values,
        )
        print(
            "positions that were imaged less than once per round",
            dfgp[dfgp["file"] < number_of_rounds].index.values,
        )

        # identify when positions that were imaged more or less than once per round
        dfgkp = df[["key", "Position"]].groupby("key").agg(lambda x: list(x))
        positions = df["Position"].unique()
        dfgkp["not_present_in_these_rounds"] = dfgkp["Position"].apply(
            lambda x: [y for y in positions if y not in x]
        )
        dfgkp["present_more_than_once_in_these_rounds"] = dfgkp["Position"].apply(
            lambda x: [
                y for y in positions if np.sum([True for xx in x if xx == y]) > 1
            ]
        )

        print(
            dfgkp[
                [
                    "not_present_in_these_rounds",
                    "present_more_than_once_in_these_rounds",
                ]
            ]
        )
        print()
