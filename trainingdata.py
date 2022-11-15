import torch

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from graph import Graph

@dataclass
class TrainingData:
    '''class to create and store training data'''
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None
    y_test = None

    def create_training_data(self, graph: Graph) -> None:
        train_indices: list = []
        train_labels:list = []

        for node, types in graph.node2types_dict.items():
            labels = [0 for _ in range(len(graph.enum_classes.keys()))]
            for t in types:
                labels[graph.enum_classes[t]] = 1.0
            train_indices.append(graph.enum_nodes[node])
            train_labels.append(labels)
        
        x_train, x_test, y_train, y_test = train_test_split(train_indices,
                                                            train_labels,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            shuffle=True) 
    
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.25,
                                                          random_state=1,
                                                          shuffle=True)

        self.x_train = torch.tensor(x_train)
        self.x_test = torch.tensor(x_test) 
        self.x_val = torch.tensor(x_val)
        self.y_val = torch.tensor(y_val)
        self.y_train = torch.tensor(y_train)
        self.y_test = torch.tensor(y_test)