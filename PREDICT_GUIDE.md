# Single Image Prediction Guide

## ✅ Problem Solved!

The dataset loader issue has been fixed! The model now automatically detects prediction mode and skips dataset loading.

## Quick Start

You can now predict knitting instructions for a single image without needing the full dataset structure!

### Method 1: Using main.py

```bash
python main.py \
  --checkpoint_dir ./checkpoint \
  --predict /path/to/your/knitting/image.png
```

### Method 2: Using standalone script

```bash
python predict_single.py \
  --image /path/to/your/knitting/image.png \
  --checkpoint_dir ./checkpoint \
  --output my_prediction.png
```

### Method 3: Test the migration

```bash
python test_migration.py
```

## What it does

1. **Detects prediction mode** - Automatically skips dataset loader when `--predict` flag is used
2. **Loads your image** and converts to grayscale
3. **Resizes to 160x160** (model input size)
4. **Normalizes pixel values** to [-0.5, 0.5] range
5. **Runs forward pass** through the neural network
6. **Saves instruction map** as PNG image
7. **Saves confidence map** showing prediction certainty

## Key Changes Made

### ✅ Fixed Dataset Loader Issue

The model constructor now checks for prediction mode:

```python
# In FeedForwardNetworks.__init__()
if not self.oparam.predict_mode:
    # Normal training/testing mode → load datasets
    self.loader = Loader(...)
else:
    # Prediction mode → skip dataset loading
    self.loader = None
    print("⚡ Skipping dataset loader, running in prediction mode")
```

### ✅ TensorFlow 2 Compatibility

- Removed all `tf.Session` dependencies
- Uses `@tf.function` and `GradientTape` for training
- Modern checkpoint management with `tf.train.Checkpoint`
- Eager execution by default

## Output Files

- `[image_name]_prediction.png` - The instruction map
- `[image_name]_prediction_confidence.png` - Confidence map (0-255 scale)

## Example Usage

```bash
# Predict instructions for a cable knit pattern
python predict_single.py \
  --image ./images/Cable2_046_16_0_back.jpg \
  --checkpoint_dir ./checkpoint \
  --output cable_instructions.png

# Use main.py with predict flag
python main.py \
  --checkpoint_dir ./checkpoint \
  --predict ./test_images/my_knit.png \
  --image_size 160
```

## Requirements

- Trained model checkpoint in `--checkpoint_dir`
- Input image (any format PIL can read: PNG, JPG, etc.)
- TensorFlow 2.8+

## Troubleshooting

### No checkpoint found
```
⚠️ No checkpoint found, using randomly initialized weights
```
**Solution**: Make sure you have a trained model in your checkpoint directory.

### Image not found
```
❌ Error: Image file not found: /path/to/image.png
```
**Solution**: Check the image path is correct and file exists.

### Model errors
If you get model-related errors, it might be because some parts of the model still need the full dataset loader. In that case, you may need to:

1. Initialize with minimal dataset structure
2. Or modify the `model_define()` method to work with single images

## Next Steps

Once you have the prediction PNG, you can:

1. **Convert to instruction table** (rows/columns of knitting instructions)
2. **Overlay on original image** for visualization
3. **Post-process** to clean up predictions
4. **Export to knitting software** formats

Would you like me to add the table conversion step as well?
