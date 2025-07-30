#!/bin/bash

FILENAME_ENV="$(pwd)/wcpo_venv"

if [ ! -d "$FILENAME_ENV" ]; then
    echo "Could not locate the virtual environment '$FILENAME_ENV'. Please re-install using:
    rm -r coarsening
    bash install.sh

If the error persists, re-pull the repository using:
    git pull git@github.com:mengsig/WCPO.git"
    exit 0
fi

echo "Activating virtual environment '$FILENAME_ENV'."
source $FILENAME_ENV/bin/activate

echo "Uncapping number of allowed threads..."
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
echo "Finished uncapping number of allowed threads..."
