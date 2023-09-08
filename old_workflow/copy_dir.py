import os
import shutil



input_dir = "/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/mip_exports_tcropped/5500000733"
output_dir = "/allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports/mip_exports_tcropped_v2/"
shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)


