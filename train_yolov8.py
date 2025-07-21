from ultralytics import YOLO
import torch
import os
import pickle
import pandas as pd

from pathlib import Path

def train_unified_model():
    # Force CUDA environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Detailed GPU check
    print("=== GPU DETECTION ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = 0  # Use first GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ERROR: No GPU detected!")
        print("Please check your CUDA installation")
        device = 'cpu'

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train with forced GPU
    results = model.train(
        data='unified_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='unified_detection',
        name='multi_class_v1',
        workers=4,  # Reduced workers
        amp=True,
        verbose=True
    )

    return model, results

def save_model_and_metrics_to_pkl(model_path=None, results_path=None, output_file='yolo_model_and_metrics.pkl'):
    """
    Save trained YOLO model and metrics to a pickle file without retraining.

    Args:
        model_path (str): Path to the trained model (.pt file). If None, uses latest from unified_detection
        results_path (str): Path to the results.csv file. If None, uses latest from unified_detection
        output_file (str): Name of the output pickle file
    """

    # Find the latest training results if paths not provided
    if model_path is None or results_path is None:
        unified_detection_path = Path('unified_detection')
        if unified_detection_path.exists():
            # Get all version directories
            version_dirs = [d for d in unified_detection_path.iterdir() if d.is_dir() and d.name.startswith('multi_class_v')]
            if version_dirs:
                # Sort by version number (extract number from multi_class_v{number})
                version_dirs.sort(key=lambda x: int(x.name.replace('multi_class_v', '')))
                latest_dir = version_dirs[-1]

                if model_path is None:
                    model_path = latest_dir / 'weights' / 'best.pt'
                if results_path is None:
                    results_path = latest_dir / 'results.csv'

                print(f"Using latest training results from: {latest_dir}")
            else:
                raise FileNotFoundError("No training results found in unified_detection directory")
        else:
            raise FileNotFoundError("unified_detection directory not found")

    # Load the trained model
    print(f"Loading model from: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(str(model_path))

    # Load metrics from results.csv
    print(f"Loading metrics from: {results_path}")
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    metrics_df = pd.read_csv(results_path)

    # Extract key metrics
    final_metrics = {
        'final_epoch': int(metrics_df.iloc[-1]['epoch']),
        'best_mAP50': float(metrics_df['metrics/mAP50(B)'].max()),
        'best_mAP50_95': float(metrics_df['metrics/mAP50-95(B)'].max()),
        'final_precision': float(metrics_df.iloc[-1]['metrics/precision(B)']),
        'final_recall': float(metrics_df.iloc[-1]['metrics/recall(B)']),
        'final_mAP50': float(metrics_df.iloc[-1]['metrics/mAP50(B)']),
        'final_mAP50_95': float(metrics_df.iloc[-1]['metrics/mAP50-95(B)']),
        'training_time_total': float(metrics_df.iloc[-1]['time']),
        'all_metrics': metrics_df.to_dict('records')  # All training history
    }

    # Get model information
    model_info = {
        'model_path': str(model_path),
        'model_type': 'YOLOv8',
        'input_size': 640,
        'classes': model.names if hasattr(model, 'names') else None,
        'num_classes': len(model.names) if hasattr(model, 'names') else None
    }

    # Create comprehensive data structure
    model_data = {
        'model': model,
        'model_info': model_info,
        'metrics': final_metrics,
        'training_args': {
            'epochs': 100,
            'batch_size': 16,
            'image_size': 640,
            'dataset': 'unified_dataset.yaml'
        }
    }

    # Save to pickle file
    print(f"Saving model and metrics to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)

    print("=" * 50)
    print("MODEL AND METRICS SAVED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Output file: {output_file}")
    print(f"Model classes: {model_info['num_classes']}")
    print(f"Final mAP@0.5: {final_metrics['final_mAP50']:.4f}")
    print(f"Best mAP@0.5: {final_metrics['best_mAP50']:.4f}")
    print(f"Final mAP@0.5:0.95: {final_metrics['final_mAP50_95']:.4f}")
    print(f"Best mAP@0.5:0.95: {final_metrics['best_mAP50_95']:.4f}")
    print(f"Training epochs: {final_metrics['final_epoch']}")
    print(f"Total training time: {final_metrics['training_time_total']:.2f} seconds")

    return model_data

def load_model_and_metrics_from_pkl(pkl_file='yolo_model_and_metrics.pkl'):
    """
    Load model and metrics from pickle file.

    Args:
        pkl_file (str): Path to the pickle file

    Returns:
        dict: Dictionary containing model, metrics, and other information
    """
    print(f"Loading model and metrics from: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        model_data = pickle.load(f)

    print("Model and metrics loaded successfully!")
    print(f"Model type: {model_data['model_info']['model_type']}")
    print(f"Number of classes: {model_data['model_info']['num_classes']}")
    print(f"Final mAP@0.5: {model_data['metrics']['final_mAP50']:.4f}")

    return model_data

if __name__ == "__main__":
    # Option 1: Train new model (original functionality)
    # model, results = train_unified_model()
    # print("Training completed!")

    # Option 2: Save existing trained model and metrics to pickle
    try:
        model_data = save_model_and_metrics_to_pkl()

        # Demonstrate loading the saved data
        print("\n" + "="*50)
        print("TESTING PICKLE LOAD...")
        print("="*50)
        loaded_data = load_model_and_metrics_from_pkl()

        # Show that the model can still be used for inference
        model = loaded_data['model']
        print(f"\nModel ready for inference!")
        print(f"Model classes: {list(model.names.values())}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained models in the unified_detection directory")



 