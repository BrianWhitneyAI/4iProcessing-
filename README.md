# multiplex_immuno_processing

[![Build Status](https://github.com/BrianWhitneyAI/multiplex_immuno_processing/workflows/Build%20Main/badge.svg)](https://github.com/BrianWhitneyAI/multiplex_immuno_processing/actions)
[![Documentation](https://github.com/BrianWhitneyAI/multiplex_immuno_processing/workflows/Documentation/badge.svg)](https://BrianWhitneyAI.github.io/multiplex_immuno_processing/)
[![Code Coverage](https://codecov.io/gh/BrianWhitneyAI/multiplex_immuno_processing/branch/main/graph/badge.svg)](https://codecov.io/gh/BrianWhitneyAI/multiplex_immuno_processing)

4i Processing contains programatic steps for imaging processing 

---
## Features

-   Store values and retain the prior value in memory
-   ... some other functionality

## Installation

**Stable Release:** `pip install multiplex_immuno_processing`<br>
**Development Head:** `pip install git+https://github.com/BrianWhitneyAI/multiplex_immuno_processing.git`

## Documentation

For full package documentation please visit [BrianWhitneyAI.github.io/multiplex_immuno_processing](https://BrianWhitneyAI.github.io/multiplex_immuno_processing).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## The Commands You Need To Know

1. `make install`

    This will setup a virtual environment local to this project and install all of the
    project's dependencies into it. The virtual env will be located in `multiplex_immuno_processing/venv`.

2. `make test`, `make fmt`, `make lint`, `make type-check`, `make import-sort`

    Quality assurance

3. `pip install -e .[dev]`

    This will install your package in editable mode with all the required development
    dependencies.

4. `make docs`

    This will generate documentation using sphinx. 

5. `make publish` and `make publish-snapshot`

    Running this command will start the process of publishing to PyPI

6. `make bumpversion' - [release, major, minor, patch, dev]
    
    update verisoning with new releases 

7. `make clean`

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean workspace.



#### Suggested Git Branch Strategy

1. `main` is for the most up-to-date development, very rarely should you directly
   commit to this branch. GitHub Actions will run on every push and on a CRON to this
   branch but still recommended to commit to your development branches and make pull
   requests to main. If you push a tagged commit with bumpversion, this will also release to PyPI.
2. Your day-to-day work should exist on branches separate from `main`. Even if it is
   just yourself working on the repository, make a PR from your working branch to `main`
   so that you can ensure your commits don't break the development head. GitHub Actions
   will run on every push to any branch or any pull request from any branch to any other
   branch.
3. It is recommended to use "Squash and Merge" commits when committing PR's. It makes
   each set of changes to `main` atomic and as a side effect naturally encourages small
   well defined PR's.



# how to run this code

### get initial configs automatically
```
python .\multiplex_immuno_processing\generate_initial_config.py --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports" --input_dirs  "\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000733" "\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000724" "\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000728" "\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000726" "\\allen\aics\microscopy\Antoine\Analyse EMT\4i Data\5500000725"
```

### manually edit config files to ensure that information is correct


### get parent image metadata for images associated with each barcode
```
python .\multiplex_immuno_processing\gather_parent_image_metadata.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```

### check the parent image metadata to determine which positions are present less or more than once per round of imaging
```
python .\multiplex_immuno_processing\check_parent_image_metadata_dataframe.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```

### use config and metadata dataframe to match all positions across rounds of imaging for each barcode
```
python .\multiplex_immuno_processing\generate_matched_position_dataframe.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```

### check the round alignment
```
python .\multiplex_immuno_processing\check_matched_position_dataframe.py  --output_path "//allen/aics/assay-dev/users/Frick/PythonProjects/Assessment/4i_testing/aligned_4i_exports"
```