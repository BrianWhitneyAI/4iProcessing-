import argparse

from glob import glob
from pathlib import Path
from glob import glob
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", type=str, required=True, help="output dir of all processing steps. This specifies where to find the yml_configs too"
)



if __name__ == "__main__":
    args = parser.parse_args()

    filedir = args.output_path + os.sep + 'pickles'

    globout = glob(filedir +os.sep + '*pickle')
    print(globout)
    for filepath in globout:
        print(Path(filepath).name)
        df = pd.read_pickle(filepath)
        dfg = df.groupby('template_position').agg('count')

        print("number of positions = ", dfg.shape[0])
        
        log = df['move_position'].values == df['template_position'].values
        print("number of position name mismatches = ", df.shape[0] - np.sum(log))
