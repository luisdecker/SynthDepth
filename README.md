# DepthTasks
Monocular depth estimation Convolutional Neural Network trained solely with synthetic data

To cite this work please use:

```
Placeholder. BibTex goes here.
```

## Requirements
The conda environment used to train this network is installable by using the environment.yml file. 

## Dataset path configuration
Change the paths on `src/default_paths.py` to fit the paths for the dataset in your system. 

## Using the model
To train the network, run `bash train_model.sh`. Configure the number of gpus in this file.

To evaluate the model in NYU and KITTI datasets, run `python src/main.py --evaluate_path path/to/model.ckpt --evaluate_config path/to/config.json`




