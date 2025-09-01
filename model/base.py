import os
from glob import glob
import tensorflow as tf
import re


class Model(object):
    """Abstract object representing an Reader model."""

    def __init__(self):
        self.vocab = None
        self.data = None

    def get_model_dir(self):
        model_dir = ""
        for attr in self._attrs:
            if hasattr(self, attr):
                model_dir += "_%s-%s" % (attr, getattr(self, attr))
        return model_dir

    def find_model_dir(self, checkpoint_dir):
        def model_dir(batch):
            dpath = ""
            for attr in self._attrs:
                if hasattr(self, attr):
                    if attr == 'batch':
                        value = batch
                    else:
                        value = getattr(self, attr)
                    dpath += "_%s-%s" % (attr, value)
            return dpath

        for i in range(1, 65):
            mdl_dir = model_dir(str(i))
            path = os.path.join(checkpoint_dir, mdl_dir)
            if os.path.exists(path):
                return mdl_dir
        return self.get_model_dir()


    def save(self, checkpoint_dir, global_step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__ or "Reader"
        model_dir = self.get_model_dir()

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # In TF2, use tf.train.Checkpoint for saving
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        if hasattr(self, 'checkpoint'):
            self.checkpoint.save(file_prefix=checkpoint_path)
        else:
            print(" [!] No checkpoint manager found. Implement checkpoint in model.")

    def load(self, checkpoint_dir, needed = False):

        # count parameters - updated for TF2
        param_count = 0
        if hasattr(self, 'trainable_variables'):
            for var in self.trainable_variables:
                if 'generator' in var.name:
                    shape = var.get_shape()
                    var_params = 1
                    for dim in shape:
                        var_params *= dim
                    param_count += var_params
        print('Generator variables: %d' % param_count)

        print(" [*] Loading checkpoints...")
        if needed:
            model_dir = self.find_model_dir(checkpoint_dir)
        else:
            model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        # TF2 checkpoint loading
        if hasattr(self, 'checkpoint'):
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print(f"Loading from: {latest_checkpoint}")
                self.checkpoint.restore(latest_checkpoint)
                print(" [*] Load SUCCESS")
                return True
            else:
                print(" [!] Load failed...")
                if needed:
                    raise FileNotFoundError(checkpoint_dir)
                return False
        else:
            print(" [!] No checkpoint manager found. Implement checkpoint in model.")
            return False
