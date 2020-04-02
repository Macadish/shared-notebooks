shared-notebooks
==============================
Shared Notebooks

_A logical, reasonably standardized, but flexible project structure for doing and sharing data science work._
#### [Project homepage](https://github.com/Macadish/cookie-cutter-compbio)
Based on the following [template.](http://drivendata.github.io/cookiecutter-data-science/)

### Requirements to use the cookiecutter template:
-----------
 - Python 2.7 or 3.5
 - [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0: This can be installed with pip by or conda depending on how you manage your Python packages:

``` bash
$ pip install cookiecutter
```

or

``` bash
$ conda config --add channels conda-forge
$ conda install cookiecutter
```

### To start a new project, run:
------------

    cookiecutter https://github.com/Macadish/cookie-cutter-compbio


[![asciicast](https://asciinema.org/a/244658.svg)](https://asciinema.org/a/244658)


### The resulting directory structure
------------

The directory structure of your new project looks like this:

```
.
├── bin                     <- Utility Functions
│   ├── jptnb.sh            <- Launch Jupyter Notebook
│   └── mkdocs.sh           <- Launch Mkdocs
├── data                    <- Store all data
│   ├── external            <- External files (i.e. databases)
│   ├── interim             <- Intermediate files
│   ├── processed           <- Processed files
│   └── raw                 <- Raw data (immutable)
├── docs                    <- Documentation (based on Mkdocs)
│   ├── docs
│   │   ├── index.md
│   │   └── misc
│   │       ├── eg1.md
│   │       ├── eg2.md
│   │       └── guide.md
│   ├── mkdocs.yml          <- Mkdocs configuration file
│   ├── references          <- Store external references, papers, pdfs etc.
│   └── README.md
├── environment             <- conda environment files
│   ├── env                 <- conda environment for python
│   ├── requirements.txt    <- pip package list for env
│   ├── md_env              <- conda environment for mkdocs
│   ├── md_requirements.txt <- pip package list for md_env
│   ├── r_env               <- conda environment for r
│   └── r_requirements.txt  <- pip package list for r_env
├── .gitignore              <- files and folders to ignore when updating git
├── README.md               <- how to navigate the folder, summary of project
├── reports                 <- outputs, figures, powerpoints, html, pdfs etc.
├── scripts                 <- scripts and notebooks
├── setup.py                <- for pip install e and import src
└── src                     <- package containing commonly used python functions
    ├── data                <- process data
    │   ├── __init__.py
    │   └── make_dataset.py
    ├── features            <- generate features
    │   ├── build_features.py
    │   └── __init__.py
    ├── __init__.py
    ├── models              <- generate models
    │   ├── __init__.py
    │   ├── predict_model.py
    │   └── train_model.py
    └── visualization       <- visualizing data
        ├── __init__.py
        └── visualize.py
```

### Installing development requirements
------------

    pip install -r environment/requirements.txt

# Mkdocs Guide
For full documentation visit [mkdocs.org](https://www.mkdocs.org).


## Commands
* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## Install conda environment
```bash
conda create --prefix ./environment/md_env python=3.7
pip install -r md_requirements.txt
```
