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

