import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class VAE:
    def __init__(self, hps, name=None):
        assert self.hparams_ok(hps)

        self.img_height = hps.img_height
        self.img_width = hps.img_width
        self.img_channels = hps.img_channels

        self.enc_blocks = hps.enc_blocks
        self.dec_blocks = hps.dec_blocks
        self.num_filters = hps.num_filters
        self.code_size = hps.code_size
        self.discrete_outputs = hps.discrete_outputs

        self.global_step = tf.train.get_or_create_global_step()
        self.scope = 'VAE' if name is None else name

        with tf.variable_scope(self.scope):
            self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            self.is_train = tf.placeholder(dtype=tf.bool, shape=[]) # required by batch norm in residual blocks
            
            batch_size = tf.shape(self.input_x)[0]

            qz = self.qz_given_x(self.input_x, training=self.is_train)
            pz = self.pz(batch_size)
            self.z = qz.sample()
            px_given_z = self.px_given_z(self.z, training=self.is_train)
            
            self.log_px_given_z = px_given_z.log_prob(self.input_x)
            self.kl_div = qz.kl_divergence(pz)

            self.elbo_x = self.log_px_given_z - self.kl_div

            self.elbo = tf.reduce_mean(self.elbo_x, axis=0)
            self.loss = -self.elbo

            self.optimizer = tf.train.AdamOptimizer(1e-3)
            tvars = [v for v in tf.trainable_variables() if v.name.startswith(self.scope)]
            self.gradients, _ = zip(*self.optimizer.compute_gradients(self.loss, tvars))

            # use control dependencies on update ops - this is required by batchnorm.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars=zip(self.gradients, tvars),
                    global_step=self.global_step)

            ## tensorboard
            tf.summary.scalar('elbo', self.elbo)
            tf.summary.scalar('reconstruction_loss', tf.reduce_mean(self.log_px_given_z))
            tf.summary.scalar('kl_div', tf.reduce_mean(self.kl_div))
            self.merged_summaries = tf.summary.merge_all()

            ## misc ops
            # decode for visualization
            self.decoded_x = px_given_z.mean()

    def pz(self, batch_size):
        mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.code_size])
        logsigma = tf.zeros(dtype=tf.float32, shape=[batch_size, self.code_size])
        z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
        z_dist = tfp.distributions.Independent(z_dist)
        return z_dist
    
    def qz_given_x(self, x, training):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            x = tf.cast(x, dtype=tf.float32)
            blocks = []
            blocks.append(x)
            for k in range(0, self.enc_blocks):
                block_input = blocks[k]
                block_output = self.encoder_block(block_input, training=training, name='block_'+str(k))
                blocks.append(block_output)

            flat = tf.layers.flatten(blocks[-1])
            fc = tf.layers.dense(flat, units=(2 * self.code_size), activation=None)
            mu, logsigma = tf.split(fc, 2, axis=1)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist
        
    def px_given_z(self, z, training):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            
            s = (2 ** self.dec_blocks)
            fc1_units = (self.img_height // s) * (self.img_width // s) * self.num_filters

            fc1 = tf.layers.dense(z, units=fc1_units, activation=None)
            decoder_input_3d = tf.reshape(
                fc1, shape=[-1, (self.img_height // s), (self.img_width // s), self.num_filters])

            blocks = []
            blocks.append(decoder_input_3d)
            for k in range(0, self.dec_blocks):
                block_input = blocks[k]
                block_output = self.decoder_block(block_input, training=training, name='block_'+str(k))
                blocks.append(block_output)

            pixel_features = blocks[-1]

            if self.discrete_outputs:
                decoded_logits_x = tf.layers.conv2d(
                    pixel_features, filters=self.img_channels, kernel_size=1, strides=1, 
                    padding='same', activation=None)
                
                x_dist = tfp.distributions.Bernoulli(logits=decoded_logits_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist
            else:
                decoded_mu_x = tf.layers.conv2d(
                    pixel_features, filters=self.img_channels, kernel_size=1, strides=1, 
                    padding='same', activation=tf.nn.tanh)

                decoded_sigma_x = 0.01 + tf.layers.conv2d(
                    pixel_features, filters=self.img_channels, kernel_size=1, strides=1, 
                    padding='same', activation=tf.nn.sigmoid)
                
                x_dist = tfp.distributions.MultivariateNormalDiag(
                    loc=decoded_mu_x, scale_diag=decoded_sigma_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist

    def encoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

            conv0 = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='valid',
                kernel_initializer=glorot_init, activation=None)

            conv1 = tf.layers.conv2d(
                conv0, filters=self.num_filters, kernel_size=3, strides=1, padding='same', 
                kernel_initializer=he_init, activation=None)
            bn1 = tf.layers.batch_normalization(conv1, training=training)
            act1 = tf.nn.relu(bn1)


            conv2 = tf.layers.conv2d(
                act1, filters=self.num_filters, kernel_size=3,strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn2 = tf.layers.batch_normalization(conv2, training=training)
            
            out = tf.nn.relu(conv0 + bn2)

            return out

    def decoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

            deconv0 = tf.layers.conv2d_transpose(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='same', 
                kernel_initializer=glorot_init, activation=None)

            deconv1 = tf.layers.conv2d(
                deconv0, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn1 = tf.layers.batch_normalization(deconv1, training=training)
            act1 = tf.nn.relu(bn1)

            deconv2 = tf.layers.conv2d(
                act1, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            bn2 = tf.layers.batch_normalization(deconv2, training=training)
            
            out = tf.nn.relu(deconv0 + bn2)

            return out

    def hparams_ok(self, hps):
        bool1 = (hps.img_height % (2 ** hps.enc_blocks) == 0)
        bool2 = (hps.img_width % (2 ** hps.enc_blocks) == 0)
        bool3 = (hps.img_height % (2 ** hps.dec_blocks) == 0)
        bool4 = (hps.img_width % (2 ** hps.dec_blocks) == 0)

        bool5a = (hps.img_channels == 1 and hps.discrete_outputs == True)  # binarized black and white
        bool5b = (hps.img_channels == 1 and hps.discrete_outputs == False) # grayscale
        bool5c = (hps.img_channels == 3 and hps.discrete_outputs == False) # color
        bool5 = (bool5a or bool5b or bool5c)

        ok = bool1 and bool2 and bool3 and bool4 and bool5
        return ok

    def train(self, sess, x):
        feed_dict = {
            self.input_x: x, 
            self.is_train: True
        }
        _ = sess.run(self.train_op, feed_dict=feed_dict)
        elbo, step, summaries = sess.run([self.elbo, self.global_step, self.merged_summaries], feed_dict=feed_dict)
        return elbo, step, summaries

    def evaluate(self, sess, x):
        feed_dict = {
            self.input_x: x,
            self.is_train: False
        }
        elbo = sess.run(self.elbo, feed_dict=feed_dict)
        return elbo

    def encode(self, sess, x):
        z = sess.run(self.z, feed_dict={self.input_x: x, self.is_train: False})
        return z

    def decode(self, sess, z):
        x_gen = sess.run(self.decoded_x, feed_dict={self.z: z, self.is_train: False})
        return x_gen

    def generate(self, sess, num_samples):
        z = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, self.code_size))
        x_gen = self.decode(sess, z)
        return x_gen

    def reconstruct(self, sess, x):
        x_re = sess.run(self.decoded_x, feed_dict={self.input_x: x, self.is_train: False})
        return x_re
