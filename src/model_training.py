import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

def train_model(data_yaml_path, model_name='yolov8n.pt', epochs=100, batch=16, imgsz=320):
    """
    Train YOLO model with error handling
    """
    try:
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load model
        model = YOLO(model_name)
        
        # Train model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch,
            device=device,
            imgsz=imgsz,
            cache=True,
            patience=10,  # Early stopping patience
            save=True,
            verbose=True
        )
        
        print("Training completed successfully!")
        return model, results
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        raise

def plot_training_results(results_dir):
    """
    Plot training metrics
    """
    try:
        # Find the latest training run
        from src.data_preprocessing import number_in_string
        log_dir = max(glob(os.path.join(results_dir, 'train*')), key=number_in_string)
        
        # Load results
        results_path = os.path.join(log_dir, 'results.csv')
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        results = pd.read_csv(results_path)
        results.columns = results.columns.str.strip()
        
        # Plot metrics
        epochs = results.index + 1
        
        plt.figure(figsize=(12, 8))
        
        # Plot mAP metrics
        plt.subplot(2, 2, 1)
        if 'metrics/mAP50(B)' in results.columns:
            map_0_5 = results['metrics/mAP50(B)']
            plt.plot(epochs, map_0_5, label='mAP@0.5', color='blue')
        
        if 'metrics/mAP50-95(B)' in results.columns:
            map_0_5_0_95 = results['metrics/mAP50-95(B)']
            plt.plot(epochs, map_0_5_0_95, label='mAP@0.5-0.95', color='red')
        
        plt.xlabel('Epochs')
        plt.ylabel('mAP')
        plt.title('mAP over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss metrics if available
        plt.subplot(2, 2, 2)
        loss_columns = [col for col in results.columns if 'loss' in col.lower()]
        for col in loss_columns[:3]:  # Plot first 3 loss metrics
            plt.plot(epochs, results[col], label=col)
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Error plotting training results: {e}")
        raise