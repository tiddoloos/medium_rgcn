# Entity type prediction with Relation Graph Convolutional Network

### Setup with conda
Clone the repository and save on the desired location on you machine.
Nevigate to the root of the respository in you terminal.
Now we setup an enviroment (e.g conda).
You can setup the environment to your linking offcourse.
This is a conda example:
```
conda create -n medium_rgcn python=3.8
conda activate medium_rgcn
```
now we have an active enviroment.
We now will install depedencies with:
```
pip install -r requirements.txt
```
The `torch_geometric` package is missing in requirements.txt.
We install this seperately as installing it with the requirements.txt could fail (don't know why).
Now Install `toch_geometric` with the following command:
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```
Note that above command could be depricated by the time you read this.
Check this page[https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html] to see the latest command fitting your pytorch version

### Run
```
python train.py
```
