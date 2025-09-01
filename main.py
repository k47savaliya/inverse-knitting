import os
import argparse
import tensorflow as tf
import numpy as np
from model import FeedForwardNetworks
from util import pp

def_learn_rate  = 0.0005
def_max_iter    = 150000
def_batch_size  = 2
def_im_size     = 160
def_threads     = def_batch_size
def_datapath    = './dataset'
def_seed        = 2018
def_model_type  = 'Forward'
def_ckpt_dir    = './checkpoint'
def_training    = True
def_weights     = ''
def_params      = ''
def_component   = ''
def_runit       = 'relu'
def_gram_layers = ''

def create_parser():
    parser = argparse.ArgumentParser(description='Neural Inverse Knitting')
    parser.add_argument("--learning_rate", type=float, default=def_learn_rate, 
                       help="Learning rate of for adam")
    parser.add_argument("--max_iter", type=int, default=def_max_iter,
                       help="The size of total iterations")
    parser.add_argument("--batch_size", type=int, default=def_batch_size,
                       help="The size of batch images")
    parser.add_argument("--image_size", type=int, default=def_im_size,
                       help="The size of width or height of image to use")
    parser.add_argument("--threads", type=int, default=def_threads,
                       help="The number of threads to use in the data pipeline")
    parser.add_argument("--dataset", type=str, default=def_datapath,
                       help="The dataset base directory")
    parser.add_argument("--seed", type=int, default=def_seed,
                       help="Random seed number")
    parser.add_argument("--model_type", type=str, default=def_model_type,
                       help="The type of model")
    parser.add_argument("--checkpoint_dir", type=str, default=def_ckpt_dir,
                       help="Directory name to save the checkpoints")
    parser.add_argument("--training", action="store_true", default=def_training,
                       help="True for training, False for testing")
    parser.add_argument("--params", nargs='*', default=def_params,
                       help="Parameter map")
    parser.add_argument("--weights", nargs='*', default=def_weights,
                       help="Weight map")
    parser.add_argument("--component", type=str, default=def_component,
                       help="Component to train (all by default), "
                            + "valid values include: transfer | norendering | warping | nuclear | none")
    parser.add_argument("--gram_layers", nargs='*', default=def_gram_layers,
                       help="List of layers for the gram loss")
    parser.add_argument("--predict", type=str, default="",
                       help="Path to a single image for prediction (bypasses dataset loader)")
    return parser

model_dict = {
    "Forward": FeedForwardNetworks
}

def main():
    # Parse arguments
    parser = create_parser()
    FLAGS = parser.parse_args()
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.checkpoint_dir + '/train'):
        os.makedirs(FLAGS.checkpoint_dir + '/train')
    if not os.path.exists(FLAGS.checkpoint_dir + '/val'):
        os.makedirs(FLAGS.checkpoint_dir + '/val')

    NNModel = model_dict[FLAGS.model_type]
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Print configuration
    for key, value in vars(FLAGS).items():
        pp.pprint([key, value])

    # object generation - no more sessions in TF2
    obj_model = NNModel(tf_flag = FLAGS)

    # Train, Test, or Predict
    if FLAGS.predict:
        # Single image prediction mode
        print(f"🎯 Single image prediction mode")
        print(f"📁 Input image: {FLAGS.predict}")
        
        # Create output path
        import os
        output_dir = FLAGS.checkpoint_dir
        image_name = os.path.splitext(os.path.basename(FLAGS.predict))[0]
        save_path = os.path.join(output_dir, f"{image_name}_prediction.png")
        
        # Run prediction
        pred_map, probs = obj_model.predict(FLAGS.predict, save_path)
        
        print(f"🎉 Prediction complete!")
        print(f"📊 Output shape: {pred_map.shape}")
        print(f"💾 Saved to: {save_path}")
        
    elif FLAGS.training:
        obj_model.train()
    else:
        obj_model.test()

if __name__ == '__main__':
    main()
