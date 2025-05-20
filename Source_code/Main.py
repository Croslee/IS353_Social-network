import torch
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Import our modules
from data_loader import download_dataset, load_and_preprocess_data, create_genre_lookup, split_data, create_user_item_graph
from models import GNNMovieLensModel
from dataset import MovieLensDataset, create_data_loaders
from evaluation import evaluate_model, find_optimal_threshold, plot_confusion_matrix, plot_training_metrics, plot_rating_distributions
from train import train_model

def main():
    start_time = time.time()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download dataset (uncomment if needed)
    # download_dataset()
    
    # Load and preprocess data
    movies, ratings, num_users, num_movies = load_and_preprocess_data(sample_percentage=0.01)
    
    # Set up genre encoding
    max_genre_count = 5  # Limit to 5 genres per movie
    genre_lookup = create_genre_lookup(movies, max_genre_count)
    
    # Split data
    train_data, test_data = split_data(ratings)
    
    # Create graph 
    edge_index, edge_attr = create_user_item_graph(train_data, num_movies, num_users)
    
    # Print dataset statistics
    print(f'num_movies: {num_movies}')
    print(f'num_users: {num_users}')
    print(f'num_genres: {max_genre_count}')
    
    # Create datasets
    train_dataset = MovieLensDataset(train_data, movies, genre_lookup, max_genre_count, num_users)
    test_dataset = MovieLensDataset(test_data, movies, genre_lookup, max_genre_count, num_users)
    
    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size=1024)
    
    # Set hyperparameters
    embedding_dim = 8
    hidden_dim = 16
    
    # Create model
    model = GNNMovieLensModel(
        num_movies=num_movies,
        num_users=num_users,
        num_genres_encoded=max_genre_count,
        embedding_size=embedding_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Train model
    model, train_losses, val_metrics_list = train_model(
        model, 
        train_loader, 
        test_loader, 
        edge_index, 
        device, 
        epochs=50, 
        stop_i=100
    )
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_metrics_list, len(train_losses))
    
    # Final evaluation
    print("Evaluating final model...")
    criterion = torch.nn.MSELoss()
    test_metrics, all_predicted_ratings, all_true_ratings, all_predicted_binary, all_true_binary = evaluate_model(
        model, test_loader, criterion, device, edge_index
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(test_metrics['tn'], test_metrics['fp'], test_metrics['fn'], test_metrics['tp'])
    
    # Find optimal threshold
    optimal_threshold, _, _, _, _ = find_optimal_threshold(all_predicted_ratings, all_true_ratings)
    
    # Re-evaluate with optimal threshold
    print(f"Re-evaluating with optimal threshold: {optimal_threshold:.2f}")
    optimal_metrics, _, _, _, _ = evaluate_model(
        model, test_loader, criterion, device, edge_index, threshold=optimal_threshold
    )
    
    # Display final metrics
    print("\nFinal Metrics with Default Threshold (3.5):")
    print(f'Test RMSE: {test_metrics["rmse"]:.3f}')
    print(f'Test Accuracy: {test_metrics["accuracy"]:.2f}%')
    print(f'Test Precision: {test_metrics["precision"]:.2f}%')
    print(f'Test Recall: {test_metrics["recall"]:.2f}%')
    print(f'Test F1 Score: {test_metrics["f1"]:.2f}%')
    
    print("\nFinal Metrics with Optimal Threshold:")
    print(f'Optimal Threshold: {optimal_threshold:.2f}')
    print(f'Test RMSE: {optimal_metrics["rmse"]:.3f}')
    print(f'Test Accuracy: {optimal_metrics["accuracy"]:.2f}%')
    print(f'Test Precision: {optimal_metrics["precision"]:.2f}%')
    print(f'Test Recall: {optimal_metrics["recall"]:.2f}%')
    print(f'Test F1 Score: {optimal_metrics["f1"]:.2f}%')
    
    # Plot rating distributions
    plot_rating_distributions(all_predicted_ratings, all_true_ratings)
    
    # Print execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.1f} seconds")

if __name__ == "__main__":
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Run the main function
    main()

def train_model(model, train_loader, test_loader, edge_index, device, epochs=50, stop_i=100):
    """Train the GNN model."""
    # Choose loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Lists to store metrics
    train_losses = []
    val_metrics_list = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0
        
        # Use tqdm for progress bar
        for i, batch in enumerate(tqdm(train_loader)):
            # Move data to device
            movie_id = batch['movie_id'].to(device)
            user_id = batch['user_id'].to(device)
            genre_id = batch['genre_id'].to(device)
            rating = batch['rating'].to(device)
            
            # Forward pass with the graph
            output = model(movie_id, user_id, genre_id, edge_index.to(device))
            loss = criterion(output, rating.unsqueeze(1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            
            if (i + 1) % 10 == 0:  # Report more frequently
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
                
            # Early stopping for debugging
            if i + 1 == stop_i:
                break
                
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)
        
        # Evaluate on validation set
        val_metrics, _, _, _, _ = evaluate_model(model, test_loader, criterion, device, edge_index)
        val_metrics_list.append(val_metrics)
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.3f}, '
              f'Validation RMSE: {val_metrics["rmse"]:.3f}, '
              f'Accuracy: {val_metrics["accuracy"]:.2f}%, '
              f'Precision: {val_metrics["precision"]:.2f}%, '
              f'Recall: {val_metrics["recall"]:.2f}%, '
              f'F1: {val_metrics["f1"]:.2f}%, '
              f'Time: {epoch_time:.1f}s')
        
        # Free up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    return model, train_losses, val_metrics_list

def evaluate_model(model, data_loader, criterion, device, edge_index, threshold=3.5):
    """Evaluate the model and compute various metrics."""
    model.eval()
    total_loss = 0.0
    all_predicted_ratings = []
    all_true_ratings = []
    all_predicted_binary = []
    all_true_binary = []

    with torch.no_grad():
        for batch in data_loader:
            movie_id = batch['movie_id'].to(device)
            user_id = batch['user_id'].to(device)
            genre_id = batch['genre_id'].to(device)
            rating = batch['rating'].to(device)

            output = model(movie_id, user_id, genre_id, edge_index.to(device))
            loss = criterion(output, rating.unsqueeze(1))
            total_loss += loss.item()

            # Store predictions and true values
            predicted_ratings = output.squeeze().cpu().numpy()
            true_ratings = rating.cpu().numpy()

            # Binary classification (liked/not liked)
            predicted_binary = (predicted_ratings >= threshold).astype(int)
            true_binary = (true_ratings >= threshold).astype(int)

            all_predicted_ratings.extend(predicted_ratings)
            all_true_ratings.extend(true_ratings)
            all_predicted_binary.extend(predicted_binary)
            all_true_binary.extend(true_binary)

    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    rmse = np.sqrt(avg_loss)

    # Classification metrics
    precision = precision_score(all_true_binary, all_predicted_binary, zero_division=0)
    recall = recall_score(all_true_binary, all_predicted_binary, zero_division=0)
    f1 = f1_score(all_true_binary, all_predicted_binary, zero_division=0)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_true_binary, all_predicted_binary, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        'loss': avg_loss,
        'rmse': rmse,
        'accuracy': accuracy * 100,  # as percentage
        'precision': precision * 100,  # as percentage
        'recall': recall * 100,  # as percentage
        'f1': f1 * 100,  # as percentage
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

    return metrics, all_predicted_ratings, all_true_ratings, all_predicted_binary, all_true_binary

def find_optimal_threshold(predicted_ratings, true_ratings):
    """Find optimal threshold for binary classification based on F1 score."""
    thresholds = np.arange(2.0, 5.1, 0.1)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        predicted_binary = (np.array(predicted_ratings) >= threshold).astype(int)
        true_binary = (np.array(true_ratings) >= 3.5).astype(int)

        precision = precision_score(true_binary, predicted_binary, zero_division=0)
        recall = recall_score(true_binary, predicted_binary, zero_division=0)
        f1 = f1_score(true_binary, predicted_binary, zero_division=0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Find threshold with highest F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Plot threshold vs metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_scores, label='Precision')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal threshold: {optimal_threshold:.1f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall and F1 Score vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_threshold, thresholds, precision_scores, recall_scores, f1_scores

def plot_confusion_matrix(tn, fp, fn, tp):
    """Plot confusion matrix."""
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_training_metrics(train_losses, val_metrics, epochs):
    """Plot training and validation metrics over epochs."""
    # Extract metrics
    val_rmse = [metrics['rmse'] for metrics in val_metrics]
    val_accuracy = [metrics['accuracy'] for metrics in val_metrics]
    val_precision = [metrics['precision'] for metrics in val_metrics]
    val_recall = [metrics['recall'] for metrics in val_metrics]
    val_f1 = [metrics['f1'] for metrics in val_metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Training loss
    axes[0, 0].plot(range(1, epochs + 1), train_losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss over Epochs')
    axes[0, 0].grid(True)

    # RMSE
    axes[0, 1].plot(range(1, epochs + 1), val_rmse)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Validation RMSE over Epochs')
    axes[0, 1].grid(True)

    # Accuracy
    axes[0, 2].plot(range(1, epochs + 1), val_accuracy)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Validation Accuracy over Epochs')
    axes[0, 2].grid(True)

    # Precision
    axes[1, 0].plot(range(1, epochs + 1), val_precision)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision (%)')
    axes[1, 0].set_title('Validation Precision over Epochs')
    axes[1, 0].grid(True)

    # Recall
    axes[1, 1].plot(range(1, epochs + 1), val_recall)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall (%)')
    axes[1, 1].set_title('Validation Recall over Epochs')
    axes[1, 1].grid(True)

    # F1 Score
    axes[1, 2].plot(range(1, epochs + 1), val_f1)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1 Score (%)')
    axes[1, 2].set_title('Validation F1 Score over Epochs')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_rating_distributions(all_predicted_ratings, all_true_ratings):
    """Plot histograms of true and predicted ratings and their differences."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_true_ratings, bins=np.arange(0, 6, 0.5), alpha=0.5, label='True ratings')
    plt.hist(all_predicted_ratings, bins=np.arange(0, 6, 0.5), alpha=0.5, label='Predicted ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Ratings')
    plt.legend()

    plt.subplot(1, 2, 2)
    errors = np.array(all_true_ratings) - np.array(all_predicted_ratings)
    plt.hist(errors, bins=np.arange(-3, 4, 0.5))
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.show()
