# This repository has been created to optimize the weighted packing problem.

## Installing the repository

If you are using **arch** linux, you can install directly via the command:

```bash
bash install.sh --IUseArchBtw
```

If you are using **debian** with the *apt* package-manager, you can install via:

```bash
bash install.sh 
```
Otherwise, if you are on another *OS* you can install as you wish. We have ensured compatability with ```python3.12```.

## Repository structure
The relevant structure of the repository follows:
```bash
.
├── env.sh
├── install.sh
└── src
    ├── algorithms
    │   ├── bho.py
    │   ├── ga.py
    │   └── pso.py
    ├── __init__.py
    ├── losses
    │   └── loss_functions.py
    ├── scripts
    │   └── main.py
    └── utils
        └── plotting_utils.py
```


Where the general purpose of this program is to run files in the *src/scripts/* directory, and leave the rest as *utils*.

## To run a sample script:
First, ensure that your environment is activated, this can be done (if *install.sh* was used) with the command:
```bash 
source env.sh
```

Then, run the following command:
```bash
python src/scripts/main.py
```
