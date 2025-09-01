import os
import sys
import time
import tensorflow as tf
import numpy as np
import scipy
from skimage.util import crop as imcrop
import re
import pdb
import pickle
from PIL import Image

from util import Loader, merge, tf_variable_summary, tf_keypoint_summary, tf_ind_to_rgb, tf_mirror_instr, tf_mirror_image, tf_summary_confusionmat, save_instr

from . import layer_modules
from . import danet
from . import basenet
from . import rendnet

from .base import Model
from .nnlib import *


# input names
REND = 'rendering'
XFER = 'transfer'
REAL = 'real'
UNSU = 'unsup'
INST_SYNT = 'instr_synt'
INST_REAL = 'instr_real'

class Parameters:
    pass


fn_clipping01 = lambda tensor: tf.fake_quant_with_min_max_args(tensor, min=0., max=1., num_bits=8)
fn_normalize_by_max = lambda tensor: tf.divide(tensor, tf.reduce_max(tensor, axis=[1,2,3], keepdims=True) + 1e-5)

def fn_loss_entropy(tensor, label):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tensor,
        labels=tf.squeeze(label, axis=[3]),
        name="entropy")

def weight_map(weights, name):
    wmap = dict()
    for token in weights:
        if len(token) > 0:
            key, val = token.split('=')
            try:
                wmap[key] = float(val)
            except ValueError:
                wmap[key] = val # probably a string
            print('%s[%s] =' % (name, key), wmap[key])
    return wmap

class FeedForwardNetworks(Model):
    """Image to Instruction Network"""

    def __init__(self, tf_flag):
        """Initialize the parameters for a network.

		Args:
		  model_type: string, 
		  batch_size: int, The size of a batch [25]
		  dataset: str, The path of dataset
		"""

        # TODO: pull out more parameters from the hard coded parameters
        # Remove sess - TF2 uses eager execution
        self.oparam = Parameters()
        self.oparam.learning_rate = tf_flag.learning_rate
        self.oparam.max_iter = tf_flag.max_iter
        self.oparam.batch_size = tf_flag.batch_size
        self.oparam.image_size = tf_flag.image_size  # ?? necessary? need to check the network architecture dependency to fixed image size..
        self.oparam.component = tf_flag.component
        self.oparam.threads = tf_flag.threads
        self.oparam.dataset = tf_flag.dataset
        self.oparam.model_type = tf_flag.model_type
        self.oparam.checkpoint_dir = tf_flag.checkpoint_dir
        self.oparam.is_train = tf_flag.training

        # set default weights
        self.oparam.weights = {
            'loss_transfer': 1.,
            'loss_syntax*' : 0.1,
            'loss_AE': 0.1,
            'loss_xentropy*': 2.,
            'loss_feedback*': 1.,
            'loss_disentgl*': 0.1,
            'loss_vgg_percept*': 0.1,
            'loss_unsup*': 0.01
        }
        self.oparam.weights.update(weight_map(tf_flag.weights, 'weights'))

        # set default parameters
        self.oparam.params = {
            'decay_rate': 0.99,
            'decay_steps': 10000,
            'augment': 1,
            'augment_src': 'best', # 0.25,
            'augment_mirror': 0, # detrimental unless we also apply in validation?
            'resi_global': 0,
            'resi_ch': 66,
            'gen_passes': 1,
            'decoder': 1,
            'discr_img': 0, 
            'discr_latent': 0, 
            'discr_instr':0,
            'discr_type': 'l2',
            'feedback': 0,
            'mean_img': 0,
            'runit': 'relu',
            'syntax_binary': 0,
            'bMILloss': 1,
            'bloss_unsup': 0, # seems detrimental
            'bloss_ae': 1, # only to be used with auto-encoder architectures
            'bloss_disentangle': 0, # seems detrimental
            'bvggloss': 0, # not always needed
            'vgg16or19': '16',
            'bg_type': 'local',
            'bg_weight': 0.1,
            'bunet_test': 0,
            'use_resnet': 0,
            'use_renderer': 0
        }
        self.oparam.params.update(weight_map(tf_flag.params, 'params'))

        # gram parameters for style supervision
        self.oparam.params['gram_layers'] = tf_flag.gram_layers
        self.oparam.params['discr'] = 1 if (self.oparam.params['discr_img'] + 
                                        self.oparam.params['discr_latent'] +
                                        self.oparam.params['discr_instr']) >= 1 else 0
        
        # register dataset path
        self.oparam.params['dataset'] = self.oparam.dataset

        # try loading parameters
        self.load_params(not(self.oparam.is_train))

        # using mean images instead of 0.5
        if self.oparam.params['mean_img']:
            print('Using mean images in %s' % os.path.join(self.oparam.dataset, 'mean'))
            mean_path = lambda name: os.path.join(self.oparam.dataset, 'mean', name)
            mean_imgs = {
                'mean_rend' : mean_path('rendering.jpg'),
                'mean_tran' : mean_path('transfer.jpg'),
                'mean_real' : mean_path('real.jpg')
            }
            self.oparam.params.update(mean_imgs)

        # special network with only encoder paths
        if not self.oparam.params.get('decoder', 1):
            self.oparam.params.update({
                # 'use_rend': 0,
                'bloss_unsup': 0,
                'bloss_ae': 0,
                'bloss_disentangle': 0,
                'bvggloss': 0,
                'resi_ch': 0
            })

        # do not use transfer data for renderer
        if self.oparam.params.get('use_renderer', 0):
            self.oparam.params['use_tran'] = 0 # no need to load transfer data

        # use rgb data for base network (for better data augmentation)
        if self.oparam.params.get('use_resnet', 0):
            self.oparam.params['xfer_type'] = 'rgb'

        # set default rectifier unit
        runit_type = self.oparam.params['runit']
        print('Rectifier unit:', runit_type)
        set_runit(runit_type)

        # parameters used to save a checkpoint
        self.lr = self.oparam.learning_rate
        self.batch = self.oparam.batch_size
        self._attrs = ['model_type', 'lr', 'batch']

        self.options = []

        self.oparam.params['training'] = self.oparam.is_train

        if self.oparam.is_train:
            self.load_params(False) # note: do not require parameters to exist at this stage
        else:
            self.oparam.params['use_tran'] = False
            self.oparam.params['use_rend'] = False
        self.loader = Loader(self.oparam.dataset, self.oparam.batch_size, self.oparam.threads, self.oparam.params)

        if len(self.loader.fakes) > 1:
            print('\n\n/!\\ Using multiple types of fake data.\nMake sure this is intended and not an error!\n', self.loader.fakes)
    
        if self.oparam.is_train:
            self.build_model()
        else:
            self.build_model_test()

    def model_define(self,
                     X_in,
                     Y_out,
                     is_train=False):

        net = dict()

        
        # [batch, height, width, channels]
        # semantic augmentation
        self.oparam.params['is_train'] = is_train
        if is_train and self.oparam.params.get('augment_mirror', 0):
            t_cond_real = tf.greater(tf.random_uniform([self.batch]), 0.5)
            t_cond_synt = tf.greater(tf.random_uniform([self.batch]), 0.5)
            for key in X_in.keys():
                # mirroring image
                t_img = X_in[key]
                if key == 'real':
                    t_cond = t_cond_real
                else:
                    t_cond = t_cond_synt
                X_in[key] = tf.where(t_cond, tf_mirror_image(t_img), t_img)
            for key in Y_out.keys():
                # mirroring instruction
                t_inst = Y_out[key]
                if key == 'real':
                    t_cond = t_cond_real
                else:
                    t_cond = t_cond_synt
                Y_out[key] = tf.where(t_cond, tf_mirror_instr(t_inst), t_inst)

        # remove unsupervised data from dictionary if not used
        if self.oparam.params.get('use_unsup', 0) == 0:
            if 'unsup' in X_in.keys():
                del X_in[UNSU]

        # model and loss
        if self.oparam.params.get('use_renderer'):
            net = rendnet.model_composited(Y_out, X_in, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = rendnet.total_loss(
                    net, X_in, self.oparam.params)
            
        elif self.oparam.params.get('use_resnet'):
            net = basenet.model_composited(X_in, Y_out, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = basenet.total_loss(
                    net, Y_out, self.oparam.params)
            
        elif self.oparam.params.get('bunet_test', 0) == 1:
            if 'rend' in X_in.keys():
                del X_in['rend']
            if 'tran' in X_in.keys():
                del X_in['tran']
            net = danet.model_composited(X_in, Y_out, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = danet.total_loss(
                    net, Y_out[INST_SYNT], Y_out[INST_REAL], self.oparam.params)
                
        elif self.oparam.params.get('bunet_test', 0) == 2: # real, rend, tran
            if self.oparam.params.get('use_cgan', 0):
                X_in['tran'] = X_in['cgan']
            net = danet.model_composited_RFI_2(X_in, Y_out, self.oparam.params)
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = danet.total_loss_RFI(
                                net, Y_out, self.oparam.params)
        elif self.oparam.params.get('bunet_test', 0) == 3: # complex net
            # if 'tran' in X_in.keys():
            #     del X_in['tran']
            net = danet.model_composited_RFI_complexnet(X_in, Y_out, self.oparam.params)
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = danet.total_loss_RFI(
                                net, Y_out, self.oparam.params)
        elif self.oparam.params.get('use_autoencoder', 0):
            net = layer_modules.model_composited(X_in, Y_out, self.oparam.params)
            
            if is_train:
                loss_dict_Disc, loss_dict_Gene, metrics = layer_modules.total_loss(
                    net, Y_out, self.oparam.params)
        else:
            raise ValueError('No model selected (use_renderer | use_resnet | bunet_test | use_autoencoder)')

        if not is_train:
            loss_dict_Disc = None
            loss_dict_Gene = None
            metrics = None
        return net, loss_dict_Disc, loss_dict_Gene, metrics

    def build_model(self):
        print('Model build')

        # @see https://www.tensorflow.org/api_docs/python/tf/data/Iterator#from_string_handle

        # iterators
        train_iter  = self.loader.iter(set_option='train')
        val_iter    = self.loader.iter(set_option='val')
        
        # handles
        self.train_handle = self.sess.run(train_iter.string_handle())
        self.val_handle   = self.sess.run(val_iter.string_handle())
        
        # create iterator switch
        self.batch_handle = tf.placeholder(tf.string, shape=[])
        batch_iter = tf.data.Iterator.from_string_handle(self.batch_handle, train_iter.output_types)

        # get effective batch
        curbatch = batch_iter.get_next()

        #import pdb
        #pdb.set_trace()
        img_size = [self.loader.batch_size, 160, 160, 1]
        lbl_size = [self.loader.batch_size, 20, 20, 1]
        # apply shapes on images and labels
        inst_synt = curbatch['synt'][-1]
        real, inst_real = curbatch['real']
        if 'unsup' in curbatch.keys():
            unsup = curbatch['unsup'][0]

        # pdb.set_trace()  # check whether unsup data structure is good enough

        # apply shapes
        for t_img in curbatch['synt'][0:-1]:
            t_img.set_shape(img_size)
        real.set_shape(img_size)
        if 'unsup' in curbatch.keys():
            unsup.set_shape(img_size)
        for t_lbl in [inst_synt, inst_real]:
            t_lbl.set_shape(lbl_size)

        self.tf_models = Parameters()
        print('Model build')
        self.tf_models.X = { REAL: real } # UNSU: unsup
        self.tf_models.Y = { INST_SYNT: inst_synt, INST_REAL: inst_real }
        # add synthetic inputs
        for i in range(len(self.loader.fakes)):
            name = self.loader.fakes[i]
            t_img = curbatch['synt'][i]
            self.tf_models.X[name] = t_img

        # replay buffer
        if self.oparam.params.get('replay_worst', 0):
            name = 'worst'
            self.tf_models.X[name] = tf.Variable(tf.ones_like(real), name = 'worst-input', dtype = tf.float32, trainable = False)
            self.tf_models.Y[name] = tf.Variable(tf.zeros_like(inst_real), name = 'worst-output', dtype = tf.int32, trainable = False)

        # Train path
        if self.oparam.is_train:
            with tf.device('/device:GPU:0'):
                self.tf_models.net, self.tf_models.loss_dict_Disc, self.tf_models.loss_dict_Gene, self.tf_models.metrics = \
                    self.model_define(
                        X_in =  self.tf_models.X,
                        Y_out = self.tf_models.Y,
                        is_train = self.oparam.is_train)
        else:
            return  # Test phase

        # dispatching global losses from name* and *name weights
        def dispatch_weights():
            new_weights = dict()
            for name, value in self.oparam.weights.items():
                if name.endswith('*'):
                    prefix = name[:-1]
                    for loss_name in self.tf_models.loss_dict_Gene.keys():
                        if loss_name.startswith(prefix):
                            new_weights[loss_name] = value
                    for loss_name in self.tf_models.loss_dict_Disc.keys():
                        if loss_name.startswith(prefix):
                            new_weights[loss_name] = value
                if name.startswith('*'):
                    suffix = name[1:]
                    for loss_name in self.tf_models.loss_dict_Gene.keys():
                        if loss_name.endswith(suffix):
                            new_weights[loss_name] = value
                    for loss_name in self.tf_models.loss_dict_Disc.keys():
                        if loss_name.endswith(suffix):
                            new_weights[loss_name] = value
            for name, value in self.oparam.weights.items():
                for loss_name in list(self.tf_models.loss_dict_Gene.keys()) + \
                                 list(self.tf_models.loss_dict_Disc.keys()):
                    if name == loss_name:
                        new_weights[name] = value

            # applying new weights
            for name, value in new_weights.items():
                self.oparam.weights[name] = value

        dispatch_weights()

        # balance loss weights when varying the amount of data
        if self.oparam.params.get('balance_weights', 1):
            if len(self.loader.fakes) == 0:
                print('Balancing weights for real data only')
                for name in self.tf_models.loss_dict_Gene.keys():
                    if name.endswith('/real'):
                        weight = self.oparam.weights.get(name, 1.0)
                        self.oparam.weights[name] = weight * 2
                        print('- %s: %f -> %f' % (name, weight, weight * 2))

        print('Losses:')
        for name in self.tf_models.loss_dict_Gene.keys():
            weight = self.oparam.weights.get(name, 1.0)
            if weight > 0:
                print('[gen] %s (%f)' % (name, weight))
        for name in self.tf_models.loss_dict_Disc.keys():
            weight = self.oparam.weights.get(name, 1.0)
            if weight > 0:
                print('[dis] %s (%f)' % (name, weight))

        # create full losses
        self.tf_models.loss_total_gene = tf.add_n([
            tf.reduce_mean(l * self.oparam.weights.get(i, 1.0)) # default weight of 1.0
            for (i, l) in self.tf_models.loss_dict_Gene.items()
        ])
        self.tf_models.loss_main_gene = tf.add_n([
            tf.reduce_mean(l * self.oparam.weights.get(i, 1.0)) # default weight of 1.0
            for (i, l) in self.tf_models.loss_dict_Gene.items()
            # filtering generator and adapter networks, and feedback
            if 'adapt' not in i and 'gen' not in i and 'feedback' not in i
        ])
        if self.oparam.params.get('discr', 1):
            self.tf_models.loss_total_disc = tf.add_n([
                tf.reduce_mean(l * self.oparam.weights.get(i, 1.0)) # default weight of 1.0
                for (i, l) in self.tf_models.loss_dict_Disc.items()
            ])
        else:
            self.tf_models.loss_total_disc = tf.constant(0)

        # summary storage
        self.summaries = {}
        net = self.tf_models.net

        # creating dictionary of images from residual dictionary
        def res_dict_imgs(res_dict, target='real', src = None):
            if src is None:
                src = target
            if src not in net.mean_imgs:
                src = 'real'
            real_dict = dict()
            if target.startswith('*'):
                for key, value in res_dict.items():
                    if key.endswith(target[1:]):
                        # real_dict[key] = net.mean_imgs[src] + value
                        real_dict[key] = value
            elif target in res_dict:
                # real_dict[target] = net.mean_imgs[src] + res_dict[target]
                real_dict[target] = res_dict[target]
            return real_dict

        use_renderer = self.oparam.params.get('use_renderer', 0)
        # visual summary
        self.summaries['images'] = dict()
        images = {
            'inputs' : net.imgs,
            'res-inps' : net.resi_imgs,
            'res-outs' : net.resi_outs,
            'ae' : res_dict_imgs(net.resi_outs, 'real'),
            'adapt' : res_dict_imgs(net.resi_outs, '*_real'),
            'generator' : res_dict_imgs(net.resi_outs, '*_gen')
        }
        for name in net.discr.keys():
            images['discr-' + name] = net.discr[name] # discriminator outputs
        for cat, data_dict in images.items():
            for name, tf_img in data_dict.items():
                sum_name = cat + '/' + name
                if cat != 'inputs' and use_renderer == 0:
                    tf_img = tf_img + 0.5
                self.summaries['images'][sum_name] = tf.summary.image(
                    sum_name, fn_clipping01(tf_img), max_outputs = 5)
        images = {
            'gt' : self.tf_models.Y,
            'outputs': dict(),
            'outputs-adapt': dict(),
            'outputs-gen': dict()
        }
        for name, t_instr in net.instr.items():
            if '_real' in name:
                images['outputs-adapt'][name] = t_instr
            elif '_gen' in name:
                images['outputs-gen'][name] = t_instr
            else:
                images['outputs'][name] = t_instr
        for cat, data_dict in images.items():
            for name, tf_img in data_dict.items():
                if 'feedback' in name:
                    sum_name = 'feedback/' + name.replace('_feedback', '')
                else:
                    sum_name = cat + '/' + name
                # label = fn_clipping01(tf_ind_to_rgb(tf_img))
                label = tf_ind_to_rgb(tf_img)
                self.summaries['images'][sum_name] = tf.summary.image(
                    sum_name, label, max_outputs = 5)

        for name, t_bg in net.bg.items():
            sum_name = 'bg/' + name
            self.summaries['images'][sum_name] = tf.summary.image(sum_name, tf.cast(t_bg, tf.float32), max_outputs = 5)

        # loss summary
        self.summaries['scalar'] = dict()
        self.summaries['scalar']['total_loss'] = tf.summary.scalar("loss_total", self.tf_models.loss_total_gene)
        for loss_name, tf_loss in dict(self.tf_models.loss_dict_Gene, **self.tf_models.loss_dict_Disc).items():
            # skip losses whose weights are disabled
            weight = self.oparam.weights.get(loss_name, 1.0)
            if weight > 0.0:
                self.summaries['scalar'][loss_name] = tf.summary.scalar(loss_name, tf.reduce_mean(tf_loss * weight))

        # metric summary
        for metric_name, tf_metric in self.tf_models.metrics.items():
            if metric_name.startswith('confusionmat'):
                self.summaries['images'][metric_name] = tf.summary.image(metric_name,
                                                            tf_summary_confusionmat(tf_metric,
                                                            numlabel=layer_modules.prog_ch,
                                                            tag=metric_name,
                                                            ), 
                                                            max_outputs = 5)
                        # tf_summary_confusionmat(tf_metric, 
                        #                     numlabel=layer_modules.prog_ch,
                        #                     tag=metric_name)
            else:
                self.summaries['scalar'][metric_name] = tf.summary.scalar(metric_name, tf_metric)

        # # gradient summary
        # t_grad_var = self.tf_models.net.fake_imgs['rend']
        # for loss_name, t_loss in self.tf_models.loss_dict_Gene.items():
        #     grad_name = 'gradient_rend2real_' + loss_name
        #     t_grad = tf.gradients(t_loss, t_grad_var)
        #     # skip if there's no contribution from that loss
        #     if t_grad[0] is None or self.oparam.weights.get(loss_name, 1.0) <= 0.0:
        #         continue
        #     grad_sum = tf_variable_summary(var = t_grad, name = grad_name)
        #     self.summaries['scalar'][grad_name] = grad_sum
        
        # activation summary
        # self.summaries['activation'] = dict()
        # for t_out in runit_list:
        #     act_name = 'activation/' + t_out.name.replace(':0', '')
        #     act_sum  = tf_variable_summary(var = t_out, name = act_name)
        #     self.summaries['activation'][act_name] = act_sum

        # _ = tf.summary.image('error_map',
        # tf.transpose(self.tf_models.loss, perm=[1,2,3,0]),
        # max_outputs=5) # Concatenate row-wise.

    def build_model_test(self):
        print('Model build')
        # iterators # handles
        test_iter  = self.loader.iter(set_option='test')
        self.test_handle = self.sess.run(test_iter.string_handle())
        
        # create iterator switch
        self.batch_handle = tf.placeholder(tf.string, shape=[])
        batch_iter = tf.data.Iterator.from_string_handle(self.batch_handle, test_iter.output_types)

        # get effective batch
        curbatch = batch_iter.get_next()

        img_size = [self.loader.batch_size, 160, 160, 1]
        lbl_size = [self.loader.batch_size, 20, 20, 1]
        # apply shapes on images and labels
        real, inst_real, self.input_names = curbatch['real']
        
        # apply shapes
        real.set_shape(img_size)
        inst_real.set_shape(lbl_size)

        self.tf_models = Parameters()
        print('Model build')
        self.tf_models.X = { REAL: real }
        self.tf_models.Y = { INST_REAL: inst_real }
        
        # Test path
        with tf.device('/device:GPU:0'):
            self.tf_models.net, _, _, _ = self.model_define(
                                        X_in = self.tf_models.X, 
                                        Y_out = self.tf_models.Y, 
                                        is_train = False)

    def train(self):
        """Train a network using TF2 eager execution"""
        
        # Initialize optimizers
        use_discr = self.oparam.params.get('discr', 1)
        
        # Create learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.oparam.learning_rate,
            decay_steps=self.oparam.params.get('decay_steps', 50000),
            decay_rate=self.oparam.params.get('decay_rate', 0.3),
            staircase=True)
        
        # Initialize optimizers with TF2 style
        self.gen_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=0.5, epsilon=1e-4)
        
        if use_discr:
            self.dis_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule * 0.5, beta_1=0.5, epsilon=1e-4)
        
        # Set up checkpoint management
        checkpoint_dir = self.oparam.checkpoint_dir
        checkpoint = tf.train.Checkpoint(
            gen_optimizer=self.gen_optimizer,
            model=self  # Save the whole model
        )
        if use_discr:
            checkpoint.dis_optimizer = self.dis_optimizer
            
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=5)
        
        # Restore checkpoint if exists
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
        
        # Set up tensorboard logging
        train_log_dir = os.path.join(checkpoint_dir, 'train')
        val_log_dir = os.path.join(checkpoint_dir, 'val')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        # Save parameters
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(self.oparam.params, f)
        
        start_time = time.time()
        global_step = 0
        
        print("Starting TF2 training...")
        
        while global_step < self.oparam.max_iter:
            try:
                # Get training batch
                train_data = self.loader.get_train_batch()
                if train_data is None:
                    break
                    
                X_in, Y_out = train_data
                
                # Training step
                gen_loss, dis_loss = self._train_step(X_in, Y_out, use_discr)
                
                global_step += 1
                
                # Validation and logging
                if global_step % 500 == 0:
                    val_data = self.loader.get_val_batch()
                    if val_data is not None:
                        val_X, val_Y = val_data
                        val_gen_loss, val_dis_loss = self._val_step(val_X, val_Y, use_discr)
                        
                        with val_summary_writer.as_default():
                            tf.summary.scalar('generator_loss', val_gen_loss, step=global_step)
                            if use_discr:
                                tf.summary.scalar('discriminator_loss', val_dis_loss, step=global_step)
                        
                        print(f"Iter: [{global_step}/{self.oparam.max_iter}] time: {time.time() - start_time:.4f}, "
                              f"vloss: [d {val_dis_loss:.4f}, g {val_gen_loss:.4f}]")
                
                # Training logging
                if global_step % 100 == 0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('generator_loss', gen_loss, step=global_step)
                        if use_discr:
                            tf.summary.scalar('discriminator_loss', dis_loss, step=global_step)
                        tf.summary.scalar('learning_rate', lr_schedule(global_step), step=global_step)
                
                # Progress logging
                if global_step % 10 == 0:
                    print(f"Iter: [{global_step}/{self.oparam.max_iter}] time: {time.time() - start_time:.4f}, "
                          f"loss: [d {dis_loss:.4f}, g {gen_loss:.4f}]")
                
                # Save checkpoint
                if global_step % 10000 == 0:
                    checkpoint_manager.save()
                    
            except Exception as e:
                print(f"Training interrupted: {e}")
                break
        
        print('Training ends.')
        checkpoint_manager.save()
        
    @tf.function
    def _train_step(self, X_in, Y_out, use_discr):
        """Single training step with gradient tape"""
        
        # Generator training
        with tf.GradientTape() as gen_tape:
            # Forward pass
            net, loss_dict_disc, loss_dict_gene, metrics = self.model_define(
                X_in=X_in, Y_out=Y_out, is_train=True)
            gen_loss = loss_dict_gene['total']
        
        # Get generator variables
        gen_vars = [var for var in self.trainable_variables 
                   if 'generator' in var.name]
        
        # Apply generator gradients
        gen_grads = gen_tape.gradient(gen_loss, gen_vars)
        self.gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))
        
        dis_loss = tf.constant(0.0)
        
        # Discriminator training (if enabled)
        if use_discr:
            with tf.GradientTape() as dis_tape:
                # Forward pass for discriminator
                net, loss_dict_disc, loss_dict_gene, metrics = self.model_define(
                    X_in=X_in, Y_out=Y_out, is_train=True)
                dis_loss = loss_dict_disc['total']
            
            # Get discriminator variables
            dis_vars = [var for var in self.trainable_variables 
                       if 'discriminator' in var.name]
            
            # Apply discriminator gradients
            dis_grads = dis_tape.gradient(dis_loss, dis_vars)
            self.dis_optimizer.apply_gradients(zip(dis_grads, dis_vars))
        
        return gen_loss, dis_loss
    
    @tf.function
    def _val_step(self, X_in, Y_out, use_discr):
        """Validation step without gradient computation"""
        
        # Forward pass only
        net, loss_dict_disc, loss_dict_gene, metrics = self.model_define(
            X_in=X_in, Y_out=Y_out, is_train=False)
        
        gen_loss = loss_dict_gene['total']
        dis_loss = loss_dict_disc['total'] if use_discr else tf.constant(0.0)
        
        return gen_loss, dis_loss

    def test_imgs(self, fnames_img, name="test_imgs"):
        pass

    def test(self, name="test"):
        """Test method using TF2 eager execution"""
        
        # Load checkpoint
        checkpoint_dir = self.oparam.checkpoint_dir
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=5)
        
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
        else:
            print("No checkpoint found, using randomly initialized weights")
        
        import cv2
        def fn_rescaleimg(x):
            x += 0.5
            x = tf.clip_by_value(x, 0.0, 1.0)
            return x * 255.0
        
        svpath = os.path.join(checkpoint_dir, 'eval')
        os.makedirs(svpath, exist_ok=True)
        
        gen_path = os.path.join(svpath, 'gen')
        os.makedirs(gen_path, exist_ok=True)
        
        cnt1 = 0
        cnt2 = 0
        
        show_info = self.oparam.params.get('show_confidence', 0)
        
        print("Starting TF2 inference...")
        
        # Process test data
        for test_batch in self.loader.get_test_data():
            try:
                X_in, names = test_batch
                
                # Forward pass for inference
                net, _, _, _ = self.model_define(X_in=X_in, Y_out={}, is_train=False)
                
                # Get outputs
                if 'real' in net.instr:
                    labels = net.instr['real']
                    logits = net.logits['real']
                    probs = tf.nn.softmax(logits)
                    
                    # Process each sample in batch
                    for i in range(tf.shape(names)[0]):
                        fname = names[i].numpy().decode('utf-8') if hasattr(names[i], 'numpy') else str(names[i])
                        
                        if show_info:
                            p = probs[i].numpy()  # p is 20x20x17
                            max_p = np.amax(p, axis=-1)
                            conf_mean = np.mean(max_p)
                            conf_std = np.std(max_p)
                            print(f'{cnt1 + 1} {fname} (conf: m={conf_mean:.6f}, s={conf_std:.6f})')
                        else:
                            print(f"\r{cnt1 + 1} {fname}", end='', flush=True)
                        
                        # Save instruction map
                        fpath = os.path.join(svpath, fname + '.png')
                        save_instr(fpath, labels[i].numpy())
                        cnt1 += 1
                        
                        # Save generated image if available
                        if 'real' in net.resi_outs:
                            regul = net.resi_outs['real']
                            fpath = os.path.join(gen_path, fname + '.png')
                            regul_img = fn_rescaleimg(regul[i]).numpy().astype(np.uint8)
                            cv2.imwrite(fpath, regul_img)
                            cnt2 += 1
                            
            except Exception as e:
                print(f"Error processing batch: {e}")
                break
        
        print('\nProcessing Done!')
        print(f'Processed {cnt1} instruction maps')
        if cnt2 > 0:
            print(f'Generated {cnt2} images')
        return
    
    def load_params(self, needed = False):
        fname = os.path.join(self.oparam.checkpoint_dir, 'params.pkl')
        try:
            with open(fname, 'rb') as f:
                new_params = pickle.load(f)
                self.oparam.params.update(new_params)
                if needed:
                    self.oparam.params['is_train'] = False
                print("Loaded parameters from %s" % fname)
                for key, value in self.oparam.params.items():
                    print('-', key, '=', value)
        except:
            if needed:
                print("[!] Error loading parameters from %s" % fname)
                raise

    def predict(self, image_path, save_path="./prediction.png"):
        """
        Predict instruction map for a single image (bypasses dataset loader)
        
        Args:
            image_path: Path to input knitting image
            save_path: Path to save the predicted instruction map
            
        Returns:
            pred_map: Predicted instruction map as numpy array
        """
        import tensorflow as tf
        from PIL import Image
        import numpy as np
        import os
        
        print(f"🔍 Predicting instructions for: {image_path}")
        
        # Load checkpoint
        checkpoint_dir = self.oparam.checkpoint_dir
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=5)
        
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"✅ Restored model from {checkpoint_manager.latest_checkpoint}")
        else:
            print("⚠️  No checkpoint found, using randomly initialized weights")
        
        # --- Load & preprocess image ---
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        original_size = img.size
        print(f"📸 Original image size: {original_size}")
        
        # Resize to model input size
        target_size = self.oparam.image_size
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        print(f"🔄 Resized to: {target_size}x{target_size}")
        
        # Convert to numpy and normalize to [-0.5, 0.5] range (typical for this model)
        img_array = np.array(img).astype(np.float32) / 255.0 - 0.5
        
        # Add batch and channel dimensions: [1, H, W, 1]
        img_tensor = np.expand_dims(img_array, axis=(0, -1))
        print(f"🔢 Input tensor shape: {img_tensor.shape}")
        
        # Convert to tensorflow tensor
        X_in = {'real': tf.constant(img_tensor)}
        Y_out = {}  # Empty for inference
        
        # --- Forward pass ---
        print("🚀 Running forward pass...")
        
        try:
            # Run model inference
            net, _, _, _ = self.model_define(X_in=X_in, Y_out=Y_out, is_train=False)
            
            # Get prediction outputs
            if 'real' in net.logits:
                logits = net.logits['real']
                probs = tf.nn.softmax(logits)
                pred_map = tf.argmax(logits, axis=-1)
                
                # Convert to numpy
                pred_map_np = pred_map[0].numpy()  # Remove batch dimension
                probs_np = probs[0].numpy()
                
                print(f"📊 Prediction map shape: {pred_map_np.shape}")
                print(f"📊 Probability shape: {probs_np.shape}")
                
                # Calculate confidence
                max_probs = np.max(probs_np, axis=-1)
                confidence = np.mean(max_probs)
                print(f"🎯 Average confidence: {confidence:.4f}")
                
            else:
                raise RuntimeError("Model output 'real' not found in logits")
                
        except Exception as e:
            print(f"❌ Error during forward pass: {e}")
            raise
        
        # --- Save prediction ---
        try:
            # Use the save_instr function if available
            from util import save_instr
            save_instr(save_path, pred_map_np)
            print(f"✅ Instruction map saved to: {save_path}")
            
        except ImportError:
            # Fallback: save as grayscale image
            # Scale prediction values to 0-255 range
            pred_img = (pred_map_np * 255 / np.max(pred_map_np)).astype(np.uint8)
            Image.fromarray(pred_img).save(save_path)
            print(f"✅ Prediction image saved to: {save_path}")
        
        # --- Optional: Save confidence map ---
        conf_path = save_path.replace('.png', '_confidence.png')
        conf_img = (max_probs * 255).astype(np.uint8)
        Image.fromarray(conf_img).save(conf_path)
        print(f"📈 Confidence map saved to: {conf_path}")
        
        return pred_map_np, probs_np
