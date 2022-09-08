import argparse
from glob import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="output dir of all processing steps. This specifies where to find the yml_configs too",
)


if __name__ == "__main__":
    args = parser.parse_args()

    filedir = r"\\allen\aics\assay-dev\users\Frick\PythonProjects\Assessment\4i_testing\aligned_4i_exports\csvs"

    filedir = args.output_path + os.sep + "csvs"
    globout = glob(filedir + os.sep + "*meta*csv")
    print(globout)
    for filepath in globout:
        print()
        print(Path(filepath).name)
        df = pd.read_csv(filepath)

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

    #     keylist = df.reset_index()['key'].unique()

    #     tplist = df.set_index('key').loc['Round 1']['template_position']
    #     mplist
    #     for key in keylist:
    #         mplist = df.set_index('key').loc[key]['move_position']

    # this is the most useful info in this dataframe
    # df[['barcode','key','Position','Scene','fname','AcquisitionTime']]
