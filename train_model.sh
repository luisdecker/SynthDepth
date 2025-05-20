# Command to initiate training of the proposed neural network

# Configuration file
CONFIG=configs/args.json

# GPUS to use for training. Configured by index, space separated
# Use GPUs 0 and 1
GPUS="0 1"

python src/main.py --config $CONFIG --gpu $GPUS

