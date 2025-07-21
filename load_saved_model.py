#!/usr/bin/env python3
"""
Script to demonstrate loading and using the saved YOLO model and metrics from pickle file.
This shows how to use the model without retraining.
"""

import pickle
import pandas as pd
from pathlib import Path

def load_and_analyze_saved_model(pkl_file='yolo_model_and_metrics.pkl'):
    """
    Load the saved model and metrics, and display comprehensive information.
    """
    print("="*60)
    print("LOADING SAVED YOLO MODEL AND METRICS")
    print("="*60)
    
    # Load the pickle file
    with open(pkl_file, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract components
    model = model_data['model']
    model_info = model_data['model_info']
    metrics = model_data['metrics']
    training_args = model_data['training_args']
    
    print(f"✓ Successfully loaded from: {pkl_file}")
    print(f"✓ File size: {Path(pkl_file).stat().st_size / (1024*1024):.2f} MB")
    
    # Display model information
    print("\n" + "="*40)
    print("MODEL INFORMATION")
    print("="*40)
    print(f"Model Type: {model_info['model_type']}")
    print(f"Input Size: {model_info['input_size']}x{model_info['input_size']}")
    print(f"Number of Classes: {model_info['num_classes']}")
    print(f"Classes: {list(model.names.values())}")
    print(f"Original Model Path: {model_info['model_path']}")
    
    # Display training configuration
    print("\n" + "="*40)
    print("TRAINING CONFIGURATION")
    print("="*40)
    for key, value in training_args.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Display performance metrics
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    print(f"Training Epochs: {metrics['final_epoch']}")
    print(f"Total Training Time: {metrics['training_time_total']/3600:.2f} hours")
    print(f"Final Precision: {metrics['final_precision']:.4f}")
    print(f"Final Recall: {metrics['final_recall']:.4f}")
    print(f"Final mAP@0.5: {metrics['final_mAP50']:.4f}")
    print(f"Final mAP@0.5:0.95: {metrics['final_mAP50_95']:.4f}")
    print(f"Best mAP@0.5: {metrics['best_mAP50']:.4f}")
    print(f"Best mAP@0.5:0.95: {metrics['best_mAP50_95']:.4f}")
    
    # Display training history summary
    print("\n" + "="*40)
    print("TRAINING HISTORY SUMMARY")
    print("="*40)
    all_metrics_df = pd.DataFrame(metrics['all_metrics'])
    print(f"Total epochs recorded: {len(all_metrics_df)}")
    print(f"mAP@0.5 progression: {all_metrics_df['metrics/mAP50(B)'].iloc[0]:.4f} → {all_metrics_df['metrics/mAP50(B)'].iloc[-1]:.4f}")
    print(f"mAP@0.5:0.95 progression: {all_metrics_df['metrics/mAP50-95(B)'].iloc[0]:.4f} → {all_metrics_df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")
    
    return model_data

def demonstrate_model_usage(model_data):
    """
    Demonstrate how to use the loaded model for inference.
    """
    print("\n" + "="*40)
    print("MODEL USAGE DEMONSTRATION")
    print("="*40)
    
    model = model_data['model']
    
    print("✓ Model is ready for inference!")
    print("✓ You can now use the model for:")
    print("  - model.predict('path/to/image.jpg')")
    print("  - model.predict('path/to/video.mp4')")
    print("  - model.val() for validation")
    print("  - model.export() to export to different formats")
    
    # Show model summary if available
    try:
        print(f"\n✓ Model architecture: {model.model}")
    except:
        print("\n✓ Model loaded successfully (architecture details not displayed)")
    
    return model

def export_metrics_to_csv(model_data, output_file='saved_model_metrics.csv'):
    """
    Export all training metrics to a CSV file for analysis.
    """
    print(f"\n📊 Exporting metrics to: {output_file}")
    
    metrics_df = pd.DataFrame(model_data['metrics']['all_metrics'])
    metrics_df.to_csv(output_file, index=False)
    
    print(f"✓ Exported {len(metrics_df)} epochs of training data")
    print(f"✓ Columns: {list(metrics_df.columns)}")
    
    return metrics_df

def main():
    """
    Main function to demonstrate loading and using saved model.
    """
    try:
        # Load the saved model and metrics
        model_data = load_and_analyze_saved_model()
        
        # Demonstrate model usage
        model = demonstrate_model_usage(model_data)
        
        # Export metrics for further analysis
        metrics_df = export_metrics_to_csv(model_data)
        
        print("\n" + "="*60)
        print("SUCCESS! Model and metrics loaded successfully.")
        print("="*60)
        print("You can now use the 'model' object for inference without retraining!")
        print("All training metrics are available in the 'model_data' dictionary.")
        
        return model_data, model
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure 'yolo_model_and_metrics.pkl' exists in the current directory.")
        return None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None

if __name__ == "__main__":
    model_data, model = main()
    
    # Example of how to use the loaded model
    if model is not None:
        print("\n" + "="*40)
        print("READY FOR INFERENCE!")
        print("="*40)
        print("Example usage:")
        print("  results = model.predict('path/to/your/image.jpg')")
        print("  results = model.predict('path/to/your/video.mp4')")
        print("  model.val()  # Run validation")
