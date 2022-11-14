import torch

from torch import nn, Tensor
from typing import List, Tuple
from sklearn.metrics import f1_score

from graph import Graph
from trainingdata import TrainingData
from model import RGCNModel
from plot import plot_results


class ModelTrainer:
    def __init__(self,
                model: nn.Module,
                epochs: int,
                lr: float,
                weight_d: float) -> None:

        self.model: nn.Module = model
        self.epochs: int = epochs
        self.lr = lr
        self.weight_d = weight_d
    

    def compute_f1(self, graph: Graph, x: Tensor, y: Tensor) -> float:
        '''evaluate the model with the F1 samples metric'''
        pred = self.model(graph.edge_index, graph.edge_type)
        pred = torch.round(pred)
        
        # f1_score function does not accept torch tensor with gradient
        pred = pred.detach().numpy()
        f1_s = f1_score(y, pred[x], average='samples', zero_division=0)
        return f1_s


    def train_model(self, graph: Graph, training_data: TrainingData) -> Tuple[List[float]]:
        '''loop to train pytorch R-GCN model'''

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_d)
        loss_f = nn.BCELoss()

        f1_ss: list = []
        losses: list = []
    
        for epoch in range(self.epochs):

            self.model.eval()
            f1_s = self.compute_f1(graph, training_data.x_val, training_data.y_val)
            f1_ss.append(f1_s)
    
            self.model.train()
            optimizer.zero_grad()
            out = self.model(graph.edge_index, graph.edge_type)
            output = loss_f(out[training_data.x_train], training_data.y_train)
            output.backward()
            optimizer.step()
            l = output.item()
            losses.append(l)
            
            # every tenth epoch print loss and F1 weighted
            if epoch%10==0:
                l = output.item()
                print(f'Epoch: {epoch}, Loss: {l:.4f}\n',
                    f'F1 weighted on validation set:{f1_s:.2f}')

        return losses, f1_ss,


if __name__=='__main__':
    file_path = './data/AIFB.nt'
    graph = Graph()
    graph.init_graph(file_path)
    graph.create_edge_data()
    graph.print_graph_statistics()

    training_data = TrainingData()
    training_data.create_training_data(graph)
    
    # define some training varibles
    emb_dim = 50
    hidden_l = 16
    epochs = 51
    lr = 0.01
    weight_d = 0.00005

    model = RGCNModel(len(graph.enum_nodes.keys()),
                        emb_dim,
                        hidden_l,
                        len(graph.enum_relations.keys()),
                        len(graph.enum_classes.keys()))

    trainer = ModelTrainer(model, epochs, lr, weight_d)
    losses, f1_ss = trainer.train_model(graph, training_data)
    
    plot_results(epochs, losses, title='BCELoss on trainig set during epochs', y_label='Loss')
    plot_results(epochs, f1_ss, title='F1 score on validation set during epochs', y_label='F1 samples')

    # evaluate model on test set and print result
    f1_s_test = trainer.compute_f1(graph, training_data.x_test, training_data.y_test)
    print(f'F1 score on test set = {f1_s_test}')
