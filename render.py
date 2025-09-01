#!/usr/bin/env python3
import argparse
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
from scipy.misc import imsave
import pdb

from model import rendnet

def create_parser():
    parser = argparse.ArgumentParser(description='Neural Inverse Knitting Renderer')
    parser.add_argument("--render_type", type=str, default="dense", help="The renderer type")
    parser.add_argument("--output_dir", type=str, default="", help="The output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to process at once")
    parser.add_argument("--noise_level", type=float, default=0.45, help="Noise level to apply on the instruction")
    parser.add_argument("inputs", nargs='*', help="Input PNG files to process")
    return parser

def read_image(fname):
    img = Image.open(fname)
    img = np.array(img)
    img = img[np.newaxis,:,:,np.newaxis].astype(np.int32)
    return img

def save_image(fname, data):
    imsave(fname, data)

def main():
    # Parse arguments
    parser = create_parser()
    FLAGS = parser.parse_args()

    # find list of inputs from positional arguments
    inputs = []
    for fname in FLAGS.inputs:
        if fname.endswith('.png'):
            if os.path.exists(fname):
                inputs.append(fname)
            else:
                print('Input %s does not exist' % fname)
                return
    # check we have something to do
    if len(inputs) == 0:
        print('No input pattern. Nothing to do!')
        return

    # parameters
    render_type = FLAGS.render_type
    batch_size  = FLAGS.batch_size
    noise_level = FLAGS.noise_level
    render_weights = render_type
    if '_' in render_type:
        render_type = render_type.split('_')[0]
    output_dir  = FLAGS.output_dir
    if output_dir == '':
        output_dir = '.'

    # Enable memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Create rendering function
    @tf.function
    def render_batch(input_batch):
        return rendnet.network(input_batch, {
            'render_type': render_type,
            'noise_level': noise_level
        })

    # load rendering network weights
    rendnet.load_weights(render_weights)

    # compute renderings by batch
    batch_start = 0
    batch_end   = len(inputs)
    while batch_start < batch_end:
        # load batch
        batch = []
        names = []
        for i in range(batch_size):
            img_idx = min(batch_end - 1, batch_start + i)
            fname = inputs[img_idx]
            img = read_image(fname)
            batch.append(img)
            names.append(fname)
        img_input = np.concatenate(batch)
        
        # Convert to tensor and compute renderings
        t_input = tf.constant(img_input, dtype=tf.int32)
        img_data = render_batch(t_input)

        # save to file
        for i in range(batch_size):
            if batch_start + i >= batch_end:
                continue
            img = np.squeeze(img_data[i, :, :, :].numpy())
            img = np.maximum(0, np.minimum(255, img * 255)).astype(np.uint8)
            fname = os.path.join(output_dir, os.path.basename(names[i]))
            save_image(fname, img)
            print('Saving %s' % fname)

        # update batch position
        batch_start += batch_size

    print('Done with %d files' % len(inputs))

if __name__ == '__main__':
    main()
