import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNMovieLensModel(nn.Module):
    def __init__(self, num_movies, num_users, num_genres_encoded, embedding_size, hidden_dim):
        super(GNNMovieLensModel, self).__init__()
        self.num_movies = num_movies
        self.num_users = num_users
        self.num_genres_encoded = num_genres_encoded

        # Node embeddings
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_embedding = nn.Embedding(num_users, embedding_size)

        # Graph Convolutional layers
        self.conv1 = GCNConv(embedding_size, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Prediction layers
        self.fc1 = nn.Linear(hidden_dim * 2 + num_genres_encoded, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, movie_id, user_id, genre_id, edge_index):
        # Get embeddings
        movie_embs = self.movie_embedding.weight
        user_embs = self.user_embedding.weight

        # Apply GNN layers to learn node representations
        x = torch.cat([movie_embs, user_embs], dim=0)  # All node embeddings

        # Apply GNN convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        # Extract representations for the specific nodes we need
        movie_gnn_emb = x[movie_id]  # GNN-enhanced movie embeddings
        user_gnn_emb = x[user_id + self.num_movies]  # GNN-enhanced user embeddings (offset by num_movies)

        # Flatten genre_id and convert to float
        genre_id = genre_id.float().view(movie_id.size(0), -1)

        # Concatenate features
        combined = torch.cat([movie_gnn_emb, user_gnn_emb, genre_id], dim=1)

        # Apply prediction layers
        out = F.relu(self.fc1(combined))
        out = self.fc2(out)

        return out
