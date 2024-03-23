# Entity type prediction with Relational Graph Convolutional Network

This repository is part of [this blog post](https://medium.com/@tls430/setup-for-entity-type-prediction-with-relational-graph-convolutional-network-pytorch-3554be0bfd5a) in Towards Data Science.

### Setup
Clone the repository and save at a desired location on you machine.
Navigate to the root of the respository in you terminal.
Now setup an enviroment.
You can setup the environment to your linking offcourse, but here is an example:
```
python -m venv venv
source venv/bin/activate
```
Now we have an active enviroment.
We install the depedencies with:
```
pip install -r requirements.txt
```

### Run
As you are still located at the root directory, start model training with:
```
python train.py
```
The programm trains the RGCN model for the `data/AIFB.nt` dataset.
It also shows plots of the loss and F1 score on the validation set during training epochs.
