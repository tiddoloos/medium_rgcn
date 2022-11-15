# Entity type prediction with Relational Graph Convolutional Network

This repository is part of [this post](https://medium.com/@tls430/setup-for-entity-type-prediction-with-relational-graph-convolutional-network-pytorch-3554be0bfd5a) on medium.com.
### Setup with conda
Clone the repository and save at a desired location on you machine.
Navigate to the root of the respository in you terminal.
Now setup an enviroment (e.g. conda).
You can setup the environment to your linking offcourse.
This is a conda example:
```
conda create -n medium_rgcn python=3.8
conda activate medium_rgcn
```
Now we have an active enviroment.
We install the depedencies with:
```
pip install -r requirements.txt
```
The `torch_geometric` package is missing in requirements.txt.
We install this seperately as installing it with the requirements.txt could fail.
Now Install `toch_geometric` in the environment with the following command:
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```
Note that above command could be depricated by the time you read this.
Check this [page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to see the latest install command for your machine.

### Run
As you are still located at the root directory, start model training with:
```
python train.py
```
The programm trains the RGCN model for the `./AIFB.nt` dataset.
It also shows plots of the loss and F1 score on the validation set during training epochs.
