# Single Image Prediction Guide

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

## What it does

1. **Loads your image** and converts to grayscale
2. **Resizes to 160x160** (model input size)
3. **Normalizes pixel values** to [-0.5, 0.5] range
4. **Runs forward pass** through the neural network
5. **Saves instruction map** as PNG image
6. **Saves confidence map** showing prediction certainty

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
