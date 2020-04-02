shared-notebooks
==============================

Shared Notebooks

Project Organization
------------
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

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# shared-notebooks
