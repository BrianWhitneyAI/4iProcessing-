import os
from pathlib import Path
import re

from aicsimageio import AICSImage
import yaml

# This will likely not work in its current format of r""
bdirlist = [
    r"\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000733 (Control wells)",
    r"\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000724",
    r"\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000728",
    r"\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000726",
    r"\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000725",
]

for bdir in bdirlist:
    barcode = Path(Path(bdir)).name

    print(barcode)
    yamd = {}
    yamd["Data"] = []
    scope_list = [x for x in os.listdir(bdir) if "ZSD" in x]
    for scope in scope_list:
        pdir = bdir + os.sep + scope
        round_list = [
            x
            for x in os.listdir(pdir)
            if bool(re.search("Time|Round [0-9]+(?!.)", x, re.IGNORECASE))
            & (Path(x).stem == Path(x).name)
        ]
        round_list = [
            x
            for x in os.listdir(pdir)
            if bool(re.search("Time|Round [0-9]+", x, re.IGNORECASE))
            & (Path(x).stem == Path(x).name)
        ]

        print(round_list)
        for round_num in round_list:
            ppath = pdir + os.sep + round_num
            czi_list = [
                x
                for x in os.listdir(ppath)
                if (".czi" in x) & bool(re.search("20x", x, re.IGNORECASE))
            ]

            for czi_name in czi_list:
                fpath = ppath + os.sep + czi_name
                fpathr = fpath.replace(os.sep, "/")
                fpath2 = '"' + fpathr + '"'
                fpath2 = fpathr
                fpath = fpath2.replace("'", "")

                # each round of imaging is a multi-scene file like with dimensions STCZYX
                # (only "time lapse rounds" have T>1).

                # All rounds have S>1 and T>1

                # define the location of the reference channel in the list of channels
                # ideally the reference channel will be the image containing the
                # nuclear fluorescence (or perhaps brightfield).

                # the reference image type (nuclear dye or brightfield) should be the consisently chosen for all rounds

                # if the image is a timelapse or round01 then choose the
                # reference channel to be the last channel in the image set , =-1.

                ref_channel = -2
                if bool(
                    re.search("time|1(?![0-9])", round_num, re.IGNORECASE)
                ):  # TODO: make sure this doesn't capture round 11
                    print("TODO: make sure this doesnt capture round 11")
                    print(round_num)
                    ref_channel = -1

                # use the parent czi file for metadata
                # find the channel names for the image file
                reader = AICSImage(fpath)
                channels = reader.channel_names

                # iterate through all scenes in metadata and look for valid scenes (i.e. scenes with image data)
                # if scene lacks image data, mark it as a scene to toss (it has no image data!)
                scenes_to_toss = []
                for scene in reader.scenes:
                    reader.set_scene(scene)
                    si = reader.current_scene_index
                    try:
                        dims = reader.dims
                    except ValueError("Future Error") as e:
                        print(str(e))
                        scenes_to_toss.append(si + 1)

                zscenes_to_toss = ",".join([str(x) for x in scenes_to_toss])
                zchannels = ",".join(channels)

                # the names for these yaml entries are weird (i.e. "iround") because they get
                # organized alphabetically and I want them to appear in a desired order.
                detaild = {
                    "iround": round_num,
                    "item": "czi",
                    "zchannels": zchannels,
                    "path": fpath,
                    "scenes_to_toss": zscenes_to_toss,
                    "ref_channel": str(channels[ref_channel]),
                }

                # subd['details'].append(detaild)
                yamd["Data"].append(detaild)
            # yamd['Data'].append(subd)
        yamd["barcode"] = barcode
        yamd["scope"] = scope

        # output path defies folder where all images get stored after they get processed
        # Should define this now or later?
        yamd[
            "output_path"
        ] = "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"

    yaml_dir = os.curdir + os.sep + "yml_configs"
    if not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir)

    yaml_path = yaml_dir + os.sep + barcode + "_initial.yml"
    print(yaml_path)
    with open(yaml_path, "w") as outfile:
        yaml.dump(yamd, outfile, default_flow_style=False)
