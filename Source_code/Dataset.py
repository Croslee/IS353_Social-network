import pandas as pd
import numpy as np
import zipfile
import gdown
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def download_dataset(url='https://drive.google.com/uc?id=1X3IpoYxAJHIBlyG6QQ_rWhSGiF5E1aL8', 
                     output='/content/dataset-ml-25m.zip'):
    """Download the MovieLens dataset from Google Drive."""
    gdown.download(url, output, quiet=False)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    
    print("Dataset downloaded and extracted successfully.")

def load_and_preprocess_data(sample_percentage=0.01):
    """Load and preprocess the MovieLens dataset."""
    print("Loading data...")
    movies = pd.read_csv('/content/ml-25m/movies.csv')
    ratings = pd.read_csv('/content/ml-25m/ratings.csv')

    # Sample the data for faster processing
    ratings = ratings.sort_values('userId')
    ratings = ratings.reset_index(drop=True)
    sample_size = int(len(ratings) * sample_percentage)
    ratings = ratings.iloc[:sample_size]

    # Clean the movies dataframe
    movies['title'] = movies['title'].str.replace('(\(\d{4}\))', '').str.strip()

    # Filter ratings to include only valid movie IDs
    valid_movie_ids = set(movies['movieId'])
    ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]

    # Fill missing values in ratings dataframe
    ratings['rating'].fillna(ratings['rating'].mean(), inplace=True)

    # Genre processing
    movies['genres'] = movies['genres'].fillna('Unknown')

    # Redefine user IDs - use simple indexing
    user_id_map = {id: i for i, id in enumerate(ratings['userId'].unique())}
    ratings['userId'] = ratings['userId'].map(user_id_map)

    # Redefine movie IDs - use simple indexing
    movie_id_map = {id: i for i, id in enumerate(movies['movieId'].unique())}
    ratings['movieId'] = ratings['movieId'].map(movie_id_map)
    movies['movieId'] = movies['movieId'].map(movie_id_map)

    # Replace NaN values with default values
    ratings = ratings.fillna(0)

    return movies, ratings, len(user_id_map), len(movie_id_map)

def create_genre_lookup(movies, max_genre_count):
    """Create a lookup dictionary for genre encodings."""
    genre_lookup = {}
    for _, row in movies.iterrows():
        movie_id = row['movieId']
        genres = row['genres'].split('|')

        # One-hot encode genres
        genre_encoding = np.zeros(max_genre_count)
        for i, genre in enumerate(genres):
            if i < max_genre_count:
                genre_encoding[i] = 1.0

        genre_lookup[movie_id] = genre_encoding
    return genre_lookup

def split_data(ratings, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(ratings, test_size=test_size, random_state=random_state)

def create_user_item_graph(ratings, num_movies, num_users):
    """Create the user-item interaction graph."""
    # Create edge indices for a bipartite graph
    user_nodes = ratings['userId'].values
    movie_nodes = ratings['movieId'].values

    # Create edges (user -> movie)
    edge_index = torch.tensor([
        np.concatenate([user_nodes + num_movies, movie_nodes]),  # Source nodes (users with offset)
        np.concatenate([movie_nodes, user_nodes + num_movies])   # Target nodes (movies)
    ], dtype=torch.long)

    # Create edge weights (ratings)
    edge_attr = torch.tensor(
        np.concatenate([ratings['rating'].values, ratings['rating'].values]),
        dtype=torch.float
    )

    return edge_index, edge_attr

class MovieLensDataset(Dataset):
    def __init__(self, data, movies, genre_lookup, max_genre_count, num_users):
        self.data = data
        self.movies = movies
        self.genre_lookup = genre_lookup  # Pre-computed genre lookup for efficiency
        self.max_genre_count = max_genre_count
        self.num_users = num_users

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        movie_id = torch.tensor(row["movieId"], dtype=torch.long)
        user_id = torch.tensor(row["userId"], dtype=torch.long)

        # Use pre-computed genre encoding
        if row['movieId'] in self.genre_lookup:
            genre_id = torch.tensor(self.genre_lookup[row['movieId']], dtype=torch.float)
        else:
            # Default encoding for unknown movies
            genre_id = torch.zeros(self.max_genre_count, dtype=torch.float)

        rating = torch.tensor(row["rating"], dtype=torch.float)
        return {"movie_id": movie_id, "user_id": user_id, "genre_id": genre_id, "rating": rating}

def create_data_loaders(train_dataset, test_dataset, batch_size=1024):
    """Create DataLoaders for training and testing."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader
