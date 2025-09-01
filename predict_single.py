#!/usr/bin/env python3
"""
Simple prediction script for Neural Inverse Knitting
Bypasses dataset loader and predicts on a single image
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from model import FeedForwardNetworks

def create_minimal_flags(checkpoint_dir, image_size=160):
    """Create minimal flags object for prediction"""
    class MinimalFlags:
        def __init__(self):
            self.learning_rate = 0.0005
            self.max_iter = 1000
            self.batch_size = 1  # Single image
            self.image_size = image_size
            self.threads = 1
            self.dataset = "./dataset"  # Won't be used
            self.seed = 2018
            self.model_type = "Forward"
            self.checkpoint_dir = checkpoint_dir
            self.training = False  # Inference mode
            self.params = []
            self.weights = []
            self.component = ""
            self.gram_layers = []
            self.predict = "dummy"  # Enable prediction mode
    
    return MinimalFlags()

def main():
    parser = argparse.ArgumentParser(description='Neural Inverse Knitting - Single Image Prediction')
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input knitting image")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing model checkpoints")
    parser.add_argument("--output", type=str, default="",
                       help="Output path for prediction (auto-generated if empty)")
    parser.add_argument("--image_size", type=int, default=160,
                       help="Input image size for the model")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ Error: Checkpoint directory not found: {args.checkpoint_dir}")
        return
    
    # Auto-generate output path if not provided
    if not args.output:
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"{image_name}_prediction.png"
    
    print("🧶 Neural Inverse Knitting - Single Image Prediction")
    print(f"📁 Input image: {args.image}")
    print(f"📂 Checkpoint dir: {args.checkpoint_dir}")
    print(f"💾 Output path: {args.output}")
    print(f"🔧 Image size: {args.image_size}x{args.image_size}")
    print("-" * 50)
    
    try:
        # Create minimal flags
        flags = create_minimal_flags(args.checkpoint_dir, args.image_size)
        
        # Initialize model
        print("🏗️  Initializing model...")
        model = FeedForwardNetworks(tf_flag=flags)
        
        # Run prediction
        print("🚀 Running prediction...")
        pred_map, probs = model.predict(args.image, args.output)
        
        print("=" * 50)
        print("🎉 Prediction completed successfully!")
        print(f"📊 Prediction shape: {pred_map.shape}")
        print(f"💾 Results saved to: {args.output}")
        
        # Print some statistics
        unique_values = np.unique(pred_map)
        print(f"🔢 Unique instruction values: {len(unique_values)}")
        print(f"📈 Value range: {unique_values.min()} - {unique_values.max()}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
