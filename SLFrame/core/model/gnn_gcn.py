import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNClient(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = None,
        dropout: float = 0.5,
        cut: str = "c1",
    ):
        super().__init__()
        self.cut = cut.lower()
        self.dropout = dropout

        if self.cut in {"c1", "c2"}:
            self.conv1 = GCNConv(in_dim, hidden_dim)
        else:
            self.conv1 = None

        if self.cut == "c2":
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv2 = None

    def forward(self, x, edge_index):
        if self.cut == "c0":
            # Client leaves all GCN computation to the server
            return x

        if self.conv1 is None:
            raise RuntimeError("conv1 is not initialized for cut=%s" % self.cut)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.cut == "c2":
            if self.conv2 is None:
                raise RuntimeError("conv2 is not initialized for cut=c2")
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GCNTail(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        cut: str = "c1",
    ):
        super().__init__()
        self.cut = cut.lower()
        self.dropout = dropout

        self.conv1 = None
        self.conv2 = None

        if self.cut == "c0":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif self.cut == "c1":
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        if self.cut == "c0":
            if self.conv1 is None or self.conv2 is None:
                raise RuntimeError("Server GCN layers expected for cut=c0")
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.cut == "c1":
            if self.conv2 is None:
                raise RuntimeError("Second GCN layer missing for cut=c1")
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # c2 skips additional GCN layers

        return self.classifier(x)
