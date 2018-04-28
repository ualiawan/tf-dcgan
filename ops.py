import tensorflow as tf

def conv2d(input, out_shape, is_train, kernel=5, stddev=0.02, leak_alpha=0.2, name="conv2d"):
	with tf.variable_scope(name):
		 w = tf.get_variable('w', [kernel, kernel, input.get_shape()[-1], out_shape], 
                              initializer=tf.random_normal_initializer(stddev=stddev))
		 b = tf.get_variable('b', [out_shape], initializer=tf.constant_initializer(0.0))
		 conv = tf.nn.conv2d(input, w, strides=[1,2,2,1], padding='SAME') + b
		 activation = tf.nn.leaky_relu(conv, leak_alpha)
		 bn = tf.contrib.layers.batch_norm(activation, is_training=is_train, center=True, scale=True,
		 	decay=0.9, updates_collections=None)
		 return bn

def deconv2d(input, out_shape,is_train=True, kernel=5, stride=2, stddev=0.02, act_fn='relu', name='deconv2d'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [kernel, kernel, out_shape[-1], input.get_shape()[-1],], 
			initializer=tf.random_normal_initializer(stddev=stddev))
		b =  tf.get_variable('b', [out_shape[-1]], initializer=tf.constant_initializer(0.0))

		deconv = tf.nn.conv2d_transpose(input, w, output_shape=out_shape,
			strides=[1, stride, stride, 1], padding='SAME')
		if act_fn == 'relu':
			activation = tf.nn.relu(deconv)
			bn = tf.contrib.layers.batch_norm(activation, is_training=is_train, center=True, scale=True,
				decay=0.9, updates_collections=None)
			return bn
		elif act_fn == 'tanh':
			activation =  tf.nn.tanh(deconv)
			return deconv

def linear(input, out_shape, stddev=0.02, name="linear"):
	with tf.variable_scope(name):
		input_shape = input.get_shape().as_list()
		m = tf.get_variable('m', [input_shape[1], out_shape],
			initializer=tf.random_normal_initializer(stddev=stddev))
		b = tf.get_variable('b', [out_shape],  initializer=tf.constant_initializer(0.0))
		
		return tf.matmul(input, m) + b
