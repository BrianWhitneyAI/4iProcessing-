import os
import numpy as np
from PIL import Image
import jinja2
import argparse
import subprocess
import os
import skimage.exposure as skex
from aicsimageio import AICSImage


parser = argparse.ArgumentParser()
parser.add_argument("--input_yaml", type=str, required=True, help="yaml config path")
parser.add_argument("--matched_position_csv_parent_dir", type=str, required=False, default=None, help="Matched position csv dirs. If this argument is given, this runs find_alignment_parameters.py ")
parser.add_argument("--matched_position_w_align_params_csv_parent_dir", type=str, required=False, default=None, help="Matched position csv with alignment parent dir. If this argument is given, this runs apply_registration.py")
parser.add_argument("--batch_process_dir", type=str, required=False, default="/allen/aics/assay-dev/users/Goutham/4iProcessing-/multiplex_immuno_processing/batch_processing", help="Where jinja tempelates are")



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)


    assert args.matched_position_csv_parent_dir is not None or args.matched_position_w_align_params_csv_parent_dir is not None, "Missing dir"
    assert not(args.matched_position_csv_parent_dir is None and args.matched_position_w_align_params_csv_parent_dir is None), "only one script can be batched at a time"

    if args.matched_position_csv_parent_dir:
        position_csv_list = [os.path.join(args.matched_position_csv_parent_dir, f) for f in os.listdir(args.matched_position_csv_parent_dir) if f.endswith(".csv") and not f.startswith(".")]
        j2env = jinja2.Environment(loader=jinja2.FileSystemLoader("/allen/aics/assay-dev/users/Goutham/4iProcessing-/multiplex_immuno_processing/batch_processing/jinja"))
    else:
        position_csv_list = [os.path.join(args.matched_position_w_align_params_csv_parent_dir, f) for f in os.listdir(args.matched_position_w_align_params_csv_parent_dir) if f.endswith(".csv") and not f.startswith(".")]
        j2env = jinja2.Environment(loader=jinja2.FileSystemLoader("/allen/aics/assay-dev/users/Goutham/4iProcessing-/multiplex_immuno_processing/batch_processing/jinja"))
    
    total_jobs = len(position_csv_list)
    for i in range(len(position_csv_list)):
        position_dir = position_csv_list[i]
        
        if args.matched_position_csv_parent_dir:
            # import pdb
            # pdb.set_trace()
            render_dict_slurm = {
            'input_yaml': args.input_yaml,
            'matched_position_csv_dir': position_dir,
            'jinja_output': os.path.join(args.batch_process_dir, "jinja_output"),
            'cwd': os.path.dirname(os.getcwd()),
            }
            template_slurm = j2env.get_template('find_alignment_params.j2')
            this_script = template_slurm.render(render_dict_slurm)
            position_outname = os.path.basename(position_csv_list[i]).split(".csv", 1)[0]
            script_path = os.path.join(args.batch_process_dir, "jinja_output",  f"find_params_position_{i}.out")  # noqa E501'
            print(script_path)
            with open(script_path, 'w') as f:
                f.writelines(this_script)
        

        else:
            # import pdb
            # pdb.set_trace()
            render_dict_slurm = {
            'input_yaml': args.input_yaml,
            'matched_position_w_align_params_csv': position_dir,
            'jinja_output': os.path.join(args.batch_process_dir, "jinja_output"),
            'cwd': os.path.dirname(os.getcwd()),
            }

            template_slurm = j2env.get_template('apply_registration.j2')
            this_script = template_slurm.render(render_dict_slurm)
            position_outname = os.path.basename(position_csv_list[i]).split(".csv", 1)[0]
            script_path = os.path.join(args.batch_process_dir, "jinja_output",  f"align_position_{i}.out")  # noqa E501'
            print(script_path)
            with open(script_path, 'w') as f:
                f.writelines(this_script)



        submission = "sbatch " + script_path
        print("Submitting command: {}".format(submission))
        process = subprocess.Popen(submission, stdout=subprocess.PIPE, shell=True)  # noqa E501
        (out, err) = process.communicate()















