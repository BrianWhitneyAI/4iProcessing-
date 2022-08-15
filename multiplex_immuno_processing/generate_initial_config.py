import argparse
import os
from pathlib import Path
import re

from aicsimageio import AICSImage
import ruamel.yaml

# run this to make the initial yaml files and then edit those files to make sure nothing is missing.
# this code only helps make the yaml files...it does not generate a perfect yaml automatically.

parser = argparse.ArgumentParser()
parser.add_argument("--input_dirs", nargs="+", default=[], help="input dirs to parse")
parser.add_argument(
    "--output_path", type=str, required=True, help="output dir of yaml file"
)
parser.add_argument(
    "--output_yaml_dir",
    type=str,
    required=True,
    help="final alignment output path to specify in yaml file",
)


def sort_rounds(rounds):
    # function to sort rounds
    Orginal_numbered_list = []
    for i in range(len(rounds)):
        Orginal_numbered_list.append(int(re.search(r"\d+", rounds[i]).group()))
    # print(f"original list is {Orginal_numbered_list}")
    sorted_list = sorted(Orginal_numbered_list)
    # print(f"sorted list is {sorted_list}")
    final_sorted_list = []
    for num in set(sorted_list):
        # index = Orginal_numbered_list.index(num)
        indexs = [i for i, val in enumerate(Orginal_numbered_list) if val == num]
        # print(f"for num {num}, indexs is {indexs}")
        for k in range(len(indexs)):
            idx = indexs[k]
            # print(idx)
            final_sorted_list.append(rounds[idx])
    return final_sorted_list


def check_regex_for_changing_ref_channel(round_name):
    # return True for 20x timelapse and round1, return False otherwise.
    # helps deal with differing regex scheming
    if bool(re.search("time", round_name, re.IGNORECASE)):
        return True
    elif bool(re.search("Round", round_name, re.IGNORECASE)):
        round_number = int(re.search(r"\d+", round_name).group())
        if round_number == 1:
            return True
        else:
            return False
    else:
        return False


def get_scenes_to_toss(reader):
    # get scenes to toss as assesed by if reader can get dimensions of that scene
    scenes_to_toss = []
    for scene in reader.scenes:
        reader.set_scene(scene)
        si = reader.current_scene_index
        try:
            _ = reader.dims
        except Exception as e:
            print(str(e))
            scenes_to_toss.append(si + 1)

    return scenes_to_toss


if __name__ == "__main__":
    args = parser.parse_args()

    for bdir in args.input_dirs:  # for each input directory
        barcode = Path(Path(bdir)).name
        print(barcode)
        config = ruamel.yaml.comments.CommentedMap()

        # config ={}
        config["Data"] = []
        scope_list = [x for x in os.listdir(bdir) if "ZSD" in x]
        for scope in scope_list:  # for each scope
            pdir = bdir + os.sep + scope
            round_list = [
                x
                for x in os.listdir(pdir)
                if bool(re.search("Time|Round [0-9]+", x, re.IGNORECASE))
                & (Path(x).stem == Path(x).name)
            ]
            round_list = sort_rounds(round_list)
            print("round list is {}".format(round_list))
            for round_num in round_list:
                ppath = os.path.join(pdir, round_num)
                Image_list = [
                    x
                    for x in os.listdir(ppath)
                    if ((".czi" in x) or (".tiff" in x))
                    & bool(re.search("20x", x, re.IGNORECASE))
                ]

                for img_name in Image_list:
                    fpath = os.path.join(ppath, img_name)
                    fpathr = fpath.replace(os.sep, "/")
                    fpath2 = '"' + fpathr + '"'
                    fpath2 = fpathr
                    fpath = fpath2.replace("'", "")

                    # each round of imaging is a multi-scene file like with dimensions STCZYX
                    # (only "time lapse rounds" have T>1).

                    # All rounds have S>1 and T>1 define the location of the reference channel in the list of channels
                    # ideally the reference channel will be the image containing
                    # the nuclear fluorescence (or perhaps brightfield)

                    # the reference image type (e.g. nuclear dye or brightfield) should
                    # be the consisently chosen for all rounds
                    # if the image is a timelapse or round01 then choose the reference
                    # channel to be the last channel in the image set , =-1.
                    if (
                        bool(re.search("time|1(?![0-9])", round_num, re.IGNORECASE))
                        and round_num != "Round 11"
                    ):
                        # print('TODO: make sure this doesnt capture round 11')
                        # print("round num is {}".format(round_num))
                        ref_channel = -1
                        # print("ref channel is {}".format(ref_channel))

                    # use the parent image file for metadata
                    # find the channel names for the image file
                    reader = AICSImage(fpath)
                    channels = reader.channel_names

                    # iterate through all scenes in metadata and look for valid scenes (i.e. scenes with image data)
                    # if scene lacks image data, mark it as a scene to toss (it has no image data!)
                    if check_regex_for_changing_ref_channel(
                        round_num
                    ):  # This should only get round1 & 20X_Timelapse
                        ref_channel = -1
                    else:
                        ref_channel = -2

                    scenes_to_toss = get_scenes_to_toss(reader)
                    zscenes_to_toss = ",".join([str(x) for x in scenes_to_toss])
                    zchannels = ",".join(channels)

                    detailid = ruamel.yaml.comments.CommentedMap()
                    detailid["round"] = round_num

                    if img_name.endswith(".czi"):
                        detailid["item"] = "czi"
                    else:
                        detailid["item"] = "tiff"

                    detailid["path"] = fpath
                    detailid["scenes_to_toss"] = zscenes_to_toss
                    detailid["ref_channel"] = str(channels[ref_channel])
                    detailid["channels"] = zchannels
                    config["Data"].append(detailid)
            config["barcode"] = barcode
            config["scope"] = scope

            # output path defines folder where all images get stored after they get processed
            config["output_path"] = args.output_yaml_dir

        output_dir = os.path.join(args.output_path, "new_yaml_output_ruml")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        out_path = os.path.join(output_dir, f"{barcode}_initial.yaml")
        # print(out_path)
        # config_out = json.dumps(config, indent=4, sort_keys=True)

        with open(out_path, "w") as outfile:
            ruamel.yaml.round_trip_dump(config, stream=outfile)
