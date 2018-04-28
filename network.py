# -*- coding: utf-8 -*-
#some parts of code are taken from https://github.com/shaohua0116/DCGAN-Tensorflow
import tensorflow as tf
from ops import *

def generator(z, is_train=True, reuse=False, batch_size=64, output_size=64, gf_dim=64, c_dim=3):

    s2, s4, s8, s16= int(output_size/2), int(output_size/4), int(output_size/8), int(output_size/16)

    with tf.variable_scope('generator', reuse=reuse):
        print('\033[93m'+scope.name+'\033[0m')
        
        h0 =  linear(z, gf_dim*8*s16*s16, name='g_0_linear')
        h0 = tf.reshape(h0, [-1, s16, s16, gf_dim*8])
        h0 = tf.contrib.layers.batch_norm(h0, scale=True, center=True, is_training=is_train,
            decay=0.9, updates_collections=None, scope="g_0_bn")
        h0 = tf.nn.relu(h0)

        h1 =  deconv2d(h0, out_shape=[batch_size, s8, s8, gf_dim*4], is_train=is_train, name='g_1_deconv')

        h2 =  deconv2d(h1, out_shape=[batch_size, s4, s4, gf_dim*2], is_train=is_train, name='g_2_deconv')

        h3 =  deconv2d(h2, out_shape=[batch_size, s2, s2, gf_dim], is_train=is_train, name='g_3_deconv')

        h4 =  deconv2d(h3, out_shape=[batch_size, output_size, output_size, c_dim], is_train=is_train, act_fn='tanh', name='g_4_deconv')
        
        return h4

def discriminator(input, is_train=True, reuse=False, batch_size=64, df_dim=64):
  
    with tf.variable_scope('discriminator', reuse=reuse):

        h0 = conv2d(input, df_dim, is_train, name="d_0_conv")
        h1 = conv2d(h0, df_dim*2, is_train, name="d_1_conv")
        h2 = conv2d(h1, df_dim*4, is_train, name="d_2_conv")
        h3 = conv2d(h2, df_dim*8, is_train, name="d_3_conv")
        h4 = linear(tf.reshape(h3, [-1, gf_dim*8*s16*s16]), 1,  name="d_4_linear")

        return tf.nn.sigmoid(h4), h4