import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GATNet model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, output_dim=128, dropout=0.2, device="gpu"):

        super(GATNet, self).__init__()

        self.gat1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        self.gat2 = GATConv(output_dim * 10, output_dim, dropout=dropout)


    def forward(self, x1, edge_index, batch):

        # encoder graph
        # return_attention_weights
        x1, weight = self.gat1(x1, edge_index,return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1, weight2 = self.gat2(x1, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = gmp(x1, batch)

        return x1, weight
