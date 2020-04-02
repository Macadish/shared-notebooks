#!/bin/bash
# Make sure conda is installed.
# Check if we are on the cluster
if [[ "$(whoami)" == "teojyj" ]];
then
source /mnt/software/unstowable/miniconda3-4.6.14/etc/profile.d/conda.sh
fi
DIR=$(dirname $0)
conda activate $DIR/../environment/env
cd $DIR/../scripts
jupyter notebook --no-browser --ip='0.0.0.0' --port=8828
