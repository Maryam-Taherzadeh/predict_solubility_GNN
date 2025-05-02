# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

# class GCNWithMolFeatures(nn.Module):
#     def __init__(self, num_node_features, num_mol_features):
#         super(GCNWithMolFeatures, self).__init__()
#         torch.manual_seed(42)

#         self.conv1 = GCNConv(num_node_features, 64)
#         self.conv2 = GCNConv(64, 64)
#         self.conv3 = GCNConv(64, 64)

#         self.graph_out_dim = 64 * 2  # Global max + mean pooling

#         self.fc1 = nn.Linear(self.graph_out_dim + num_mol_features, 64)
#         self.fc2 = nn.Linear(64, 1)

#     def forward(self, x, edge_index, batch, mol_features):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.relu(self.conv3(x, edge_index))

#         pooled = torch.cat([
#             global_max_pool(x, batch),
#             global_mean_pool(x, batch)
#         ], dim=1)

#         combined = torch.cat([pooled, mol_features], dim=1)

#         combined = F.relu(self.fc1(combined))
#         out = self.fc2(combined)

#         return out.squeeze()
    

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class GCNWithMolFeatures(torch.nn.Module):
    def __init__(self, num_node_features, num_mol_features):
        super(GCNWithMolFeatures, self).__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.graph_out_dim = 64 * 2  # global max + mean pooling

        self.fc1 = torch.nn.Linear(self.graph_out_dim + num_mol_features, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, batch, mol_features):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        pooled = torch.cat([
            global_max_pool(x, batch),
            global_mean_pool(x, batch)
        ], dim=1)

        combined = torch.cat([pooled, mol_features], dim=1)
        combined = F.relu(self.fc1(combined))
        out = self.fc2(combined)

        return out.squeeze()

    
