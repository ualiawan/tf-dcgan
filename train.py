import tensorflow as tf
import time
import network
import sys
import os
import numpy as np
from glob import glob
from random import shuffle
from utils import *


flags = tf.app.flags
flags.DEFINE_integer("epochs", 25, "Number of epochs")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam optimizer")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam optimizer")
flags.DEFINE_integer("train_size", np.inf, "The size of train images")
flags.DEFINE_integer("batch_size", 64, "The number of batch images")
flags.DEFINE_integer("image_size", 108, "The size of image to use")
flags.DEFINE_integer("output_size", 64, "The size of the output images")
flags.DEFINE_integer("sample_size", 64, "The number of sample images")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color")
flags.DEFINE_integer("summary_step", 500, "The interval of generating summary")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints")
flags.DEFINE_string("dataset", "celebA", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints")
flags.DEFINE_string("summaries_dir", "logs", "Directory name to save the logs")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing")
FLAGS = flags.FLAGS


def main(_):
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.summaries_dir):
        os.makedirs(FLAGS.summaries_dir)
        
    with tf.device("/gpu:0"):
    #with tf.device("/cpu:0"):
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name="g_input_noise")
        x =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='d_input_images')
        
        Gz =  network.generator(z)
        Dx, Dx_logits =  network.discriminator(x)
        Dz, Dz_logits = network.discriminator(Gz, reuse=True)
        Gz1 =  network.generator(z, is_train=False, reuse=True)
        
        tf.summary.histogram("d_real", self.D)
        tf.summary.histogram("d_noise", self.Dz)
        tf.summary.image("G", Gz1, 10)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx_logits, labels=tf.ones_like(Dx)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dz_logits, labels=tf.zeros_like(Dz)))
        d_loss =  d_loss_real + d_loss_fake
        
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dz_logits, labels=tf.ones_like(Dz)))
        

        tf.summary.scalar("generator_loss", g_loss)
        tf.summary.scalar("discriminator_loss", d_loss)
        tf.summary.scalar("discriminator_loss_real", d_loss_real)
        tf.summary.scalar("discriminator_loss_fake", d_loss_fake)
        
        tvars = tf.trainable_variables()
        d_vars =  [var for var in tvars if 'd_' in var.name]
        g_vars =  [var for var in tvars if 'g_' in var.name]

        print(d_vars)
        print("---------------")
        print(g_vars)
        
        with tf.variable_scope(tf.get_variable_scope(),reuse=False): 
            print("reuse or not: {}".format(tf.get_variable_scope().reuse))
            assert tf.get_variable_scope().reuse == False, "Houston tengo un problem"
            d_trainer = tf.train.AdamOptimizer(FLAGS.learning_rate, FLAGS.beta1).minimize(d_loss, var_list=d_vars)
            g_trainer = tf.train.AdamOptimizer(FLAGS.learning_rate, FLAGS.beta1).minimize(g_loss, var_list=g_vars)
        

        merged = tf.summary.merge_all()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
        gpu_options.allow_growth = True
              
        saver = tf.train.Saver(max_to_keep=100)
        pretrain_saver = tf.train.Saver(var_list=tvars, max_to_keep=1)
        
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        
        print("starting session")
        summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        
        data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
        
        model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
        save_path = os.path.join(FLAGS.checkpoint_dir, model_dir)
        pretrain_saver_path = os.path.join(FLAGS.checkpoint_dir, model_dir+"_pretrain")

        iteration = 0
        print("---------------------------------------------")
        print("*************training starting***************")
        print("---------------------------------------------")
        for epoch in range(FLAGS.epoch):

            d_total_cost = 0.
            g_total_cost = 0.
            shuffle(data_files)
            num_batches = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
            #num_batches = 2
            for batch_i in range(num_batches):
                batch_files = data_files[batch_i*FLAGS.batch_size:(batch_i+1)*FLAGS.batch_size]
                batch = [load_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size) for batch_file in batch_files]
                batch_x = np.array(batch).astype(np.float32)
                batch_z = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
                start_time = time.time()
                
                d_err, _ = sess.run([d_loss, d_trainer], feed_dict={z: batch_z, x: batch_x})
                g_err, _ = sess.run([g_loss, g_trainer], feed_dict={z: batch_z})
                g_err1, _ = sess.run([g_loss, g_trainer], feed_dict={z: batch_z})

                iteration += 1
                
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                	epoch, batch_i, num_batches, time.time() - start_time, d_err, g_err1))


                if np.mod(iteration, FLAGS.summary_step) == 0:
                	summary = sess.run(merged, feed_dict={x: batch_x, z: batch_z})
                	summary_writer.add_summary(summary, iteration)

                if np.mod(iteration, FLAGS.save_step) == 0:
	                # save current network parameters
	                print("[*] Saving checkpoints...")
	                pretrain_saver.save(sess, pretrain_saver_path)
	                print("[*] Saving checkpoints SUCCESS!")

            sys.stdout.flush()
        save_path = saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)
        sys.stdout.flush()
    sess.close()   

if __name__ == '__main__':
    tf.app.run()