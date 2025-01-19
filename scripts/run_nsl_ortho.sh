#!/bin/bash

source /etc/profile
module load anaconda/2023a-pytorch

echo "My task ID:" $LLSUB_RANK
echo "Number of Tasks:" $LLSUB_SIZE

nohup python run_nsl_ortho.py $LLSUB_RANK $LLSUB_SIZE
