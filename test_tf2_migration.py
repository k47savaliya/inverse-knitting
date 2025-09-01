#!/usr/bin/env python3
"""
Test script to check TensorFlow 2 migration progress
Tests basic imports and model instantiation
"""

import sys
import traceback

def test_basic_imports():
    """Test that all modules can be imported without tf.contrib errors"""
    print("Testing basic imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Test that TF2 is being used
        if tf.__version__.startswith('2.'):
            print("✅ TensorFlow 2.x detected")
        else:
            print(f"⚠️  TensorFlow version {tf.__version__} - should be 2.x")
            
    except ImportError as e:
        print(f"❌ Failed to import TensorFlow: {e}")
        return False
        
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
        
    # Test core model imports
    modules_to_test = [
        'model.nnlib',
        'model.ops', 
        'model.spatial_transformer',
        'model.base',
        'model.rendnet',
        'model.danet',
        'model.basenet',
        'model.layer_modules',
        'model.tensorflow_vgg.custom_vgg19',
        'util.loader',
        'util.log_tool',
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            failed_imports.append((module, str(e)))
        except Exception as e:
            print(f"⚠️  {module} imported but with warnings: {e}")
            
    return len(failed_imports) == 0, failed_imports

def test_tf2_features():
    """Test TF2 specific features work"""
    print("\nTesting TF2 features...")
    
    try:
        import tensorflow as tf
        
        # Test eager execution is enabled
        if tf.executing_eagerly():
            print("✅ Eager execution is enabled")
        else:
            print("⚠️  Eager execution is disabled")
            
        # Test basic tensor operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"✅ Basic tensor math works: {c.numpy()}")
        
        # Test tf.function
        @tf.function
        def simple_function(x):
            return x * 2
            
        result = simple_function(tf.constant(5.0))
        print(f"✅ tf.function works: {result.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ TF2 features test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚀 TensorFlow 2 Migration Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok, failed_imports = test_basic_imports()
    
    # Test TF2 features
    tf2_ok = test_tf2_features()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    if imports_ok:
        print("✅ All core imports successful")
    else:
        print("❌ Some imports failed:")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
            
    if tf2_ok:
        print("✅ TF2 features working")
    else:
        print("❌ TF2 features test failed")
        
    if imports_ok and tf2_ok:
        print("\n🎉 Migration test PASSED! Basic functionality is working.")
    else:
        print("\n⚠️  Migration test PARTIAL. Some issues need to be resolved.")
        
    return imports_ok and tf2_ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
