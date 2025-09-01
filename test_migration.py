#!/usr/bin/env python3
"""
Test script to verify TF2 migration and prediction mode works
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np

# Add current directory to path so we can import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_init():
    """Test that model initializes correctly in prediction mode"""
    
    print("🧪 Testing model initialization in prediction mode...")
    
    # Create test flags
    class TestFlags:
        def __init__(self):
            self.learning_rate = 0.0005
            self.max_iter = 1000
            self.batch_size = 1
            self.image_size = 160
            self.threads = 1
            self.dataset = "./test_dataset"  # Doesn't need to exist
            self.seed = 2018
            self.model_type = "Forward"
            self.checkpoint_dir = "./test_checkpoint"
            self.training = False
            self.params = []
            self.weights = []
            self.component = ""
            self.gram_layers = []
            self.predict = "test_image.png"  # Enable prediction mode
    
    try:
        # Test model import
        print("📦 Importing model...")
        from model import FeedForwardNetworks
        
        # Test model initialization
        print("🏗️  Initializing model in prediction mode...")
        flags = TestFlags()
        model = FeedForwardNetworks(tf_flag=flags)
        
        print("✅ Model initialization successful!")
        print(f"   - Prediction mode: {model.oparam.predict_mode}")
        print(f"   - Loader: {model.loader}")
        print(f"   - Image size: {model.oparam.image_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tf2_features():
    """Test TF2 features are working"""
    
    print("\n🧪 Testing TensorFlow 2 features...")
    
    try:
        # Test eager execution
        print("⚡ Testing eager execution...")
        x = tf.constant([1, 2, 3, 4])
        y = x * 2
        print(f"   Result: {y.numpy()}")
        
        # Test tf.function
        print("🚀 Testing @tf.function...")
        @tf.function
        def test_function(x):
            return tf.reduce_sum(x)
        
        result = test_function(tf.constant([1, 2, 3]))
        print(f"   Result: {result.numpy()}")
        
        # Test gradient tape
        print("📊 Testing GradientTape...")
        x = tf.Variable(3.0)
        with tf.GradientTape() as tape:
            y = x ** 2
        
        dy_dx = tape.gradient(y, x)
        print(f"   Gradient: {dy_dx.numpy()}")
        
        print("✅ TensorFlow 2 features working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧶 Neural Inverse Knitting - TF2 Migration Test")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"🔧 TensorFlow version: {tf.__version__}")
    
    # Test TF2 features
    tf2_ok = test_tf2_features()
    
    # Test model initialization
    model_ok = test_model_init()
    
    print("\n" + "=" * 50)
    if tf2_ok and model_ok:
        print("🎉 All tests passed! TF2 migration successful!")
        print("\n✅ You can now run:")
        print("   python main.py --predict your_image.png --checkpoint_dir ./checkpoint")
        print("   python predict_single.py --image your_image.png --checkpoint_dir ./checkpoint")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
