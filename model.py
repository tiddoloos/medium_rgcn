import torch

from torch import nn
from torch import Tensor
from torch_geometric.nn import RGCNConv


class RGCNModel(nn.Module):
    def __init__ (self, num_nodes, emb_dim, hidden_l, num_relations, num_classes) -> None:
        super(RGCNModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.rgcn1 = RGCNConv(in_channels=emb_dim, out_channels=hidden_l, num_relations=num_relations, num_bases=None)
        self.rgcn2 = RGCNConv(in_channels=hidden_l, out_channels=num_classes, num_relations=num_relations, num_bases=None)

        # intialize weights
        nn.init.kaiming_uniform_(self.rgcn1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.rgcn2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, edge_index, edge_type) -> Tensor:
        x = self.rgcn1(self.embedding.weight, edge_index, edge_type)
        x = torch.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        x = torch.sigmoid(x)
        return x