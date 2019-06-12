import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class VAE:
    def __init__(self, hps, name=None):
        self.img_height = hps.img_height
        self.img_width = hps.img_width
        self.img_channels = hps.img_channels
        
        self.z_dim = hps.z_dim
        self.num_filters = 32
        self.activation = self.get_activation(hps.activation)
        self.discrete_outputs = hps.discrete_outputs

        self.global_step = tf.train.get_or_create_global_step()

        self.scope = 'VAE' if name is None else name

        with tf.variable_scope(self.scope):
            self.input_x = tf.placeholder(
                dtype=tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            
            qz = self.qz_given_x(self.input_x)
            self.z = qz.sample()
            
            decoded_px = self.px_given_z(self.z)
            pz = self.pz(batch_size=tf.shape(self.input_x)[0])
            
            self.log_px_given_z = decoded_px.log_prob(self.input_x)
            self.kl_div = qz.kl_divergence(pz)

            self.elbo_x = self.log_px_given_z - self.kl_div

            self.elbo = tf.reduce_mean(self.elbo_x, axis=0)
            self.loss = -self.elbo

            self.optimizer = tf.train.AdamOptimizer(1e-4)
            tvars = [v for v in tf.trainable_variables() if v.name.startswith(self.scope)]
            self.gradients, _ = zip(*self.optimizer.compute_gradients(self.loss, tvars))
            self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 10.0)
            self.train_op = self.optimizer.apply_gradients(
                grads_and_vars=zip(self.clipped_gradients, tvars),
                global_step=self.global_step)

            ## tensorboard
            tf.summary.scalar('elbo', self.elbo)
            tf.summary.scalar('kl_div', tf.reduce_mean(self.kl_div))
            self.merged_summaries = tf.summary.merge_all()

            ## misc ops
            # decode for visualization
            self.decoded_x = decoded_px.mean()

    def pz(self, batch_size):
        mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.z_dim])
        logsigma = tf.zeros(dtype=tf.float32, shape=[batch_size, self.z_dim])
        z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
        z_dist = tfp.distributions.Independent(z_dist)
        return z_dist
    
    def qz_given_x(self, x):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            x = tf.cast(x, dtype=tf.float32)
            r1 = self.encoder_res_block(x, name='block1')
            r2 = self.encoder_res_block(r1, name='block2')
            r3 = self.encoder_res_block(r2, name='block3')
            flat = tf.layers.flatten(r3)
            fc = tf.layers.dense(flat, units=(2 * self.z_dim), activation=None)
            mu, logsigma = tf.split(fc, 2, axis=1)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist
        
    def px_given_z(self, z):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            
            s = (2 ** 3)
            fc1_units = (self.img_height // s) * (self.img_width // s) * self.num_filters

            fc1 = tf.layers.dense(z, units=fc1_units, activation=None)
            decoder_input_3d = tf.reshape(
                fc1, shape=[-1, (self.img_height // s), (self.img_width // s), self.num_filters])

            r1 = self.decoder_res_block(decoder_input_3d, name='block1')
            r2 = self.decoder_res_block(r1, name='block2')
            r3 = self.decoder_res_block(r2, name='block3')
            res_tower_output = r3

            if self.discrete_outputs:
                decoded_logits_x = tf.layers.conv2d_transpose(
                    res_tower_output, filters=self.img_channels, kernel_size=1, strides=1, 
                    padding='same', activation=None)
                
                x_dist = tfp.distributions.Bernoulli(logits=decoded_logits_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist
            else:
                decoded_mu_x = tf.layers.conv2d_transpose(
                    res_tower_output, filters=self.img_channels, kernel_size=1, strides=1, 
                    padding='same', activation=tf.nn.tanh)

                decoded_sigma_x = 0.10 + 1.0 + tf.layers.conv2d_transpose(
                    res_tower_output, filters=self.img_channels, kernel_size=1, strides=1, 
                    padding='same', activation=tf.nn.elu)
                
                x_dist = tfp.distributions.MultivariateNormalDiag(
                    loc=decoded_mu_x, scale_diag=decoded_sigma_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist

    def encoder_res_block(self, inputs, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            c0 = tf.layers.conv2d(inputs, filters=self.num_filters, kernel_size=4, strides=2, 
                padding='valid', activation=self.activation)

            c1 = tf.layers.conv2d(c0, filters=self.num_filters, kernel_size=4, strides=1, 
                padding='same', activation=self.activation)

            c2 = tf.layers.conv2d(c1, filters=self.num_filters, kernel_size=4, strides=1, 
                padding='same', activation=self.activation)

            return c0 + c2

    def decoder_res_block(self, inputs, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            d0 = tf.layers.conv2d_transpose(inputs, filters=self.num_filters, kernel_size=4, strides=2, 
                padding='same', activation=self.activation)

            d1 = tf.layers.conv2d_transpose(d0, filters=self.num_filters, kernel_size=4, strides=1, 
                padding='same', activation=self.activation)

            d2 = tf.layers.conv2d_transpose(d1, filters=self.num_filters, kernel_size=4, strides=1, 
                padding='same', activation=self.activation)

            return d0 + d2

    def get_activation(self, name):
        activations = {
            'relu': tf.nn.relu,
            'elu': tf.nn.elu
        }
        return activations[name]

    def train(self, sess, x):
        _, elbo, gs, ms = sess.run(
            [self.train_op, self.elbo, self.global_step, self.merged_summaries], feed_dict={self.input_x: x})
        return elbo, gs, ms

    def evaluate(self, sess, x):
        elbo = sess.run(self.elbo, feed_dict={self.input_x: x})
        return elbo

    def reconstruct(self, sess, x):
        x_recon = sess.run(self.decoded_x, feed_dict={self.input_x: x})
        return x_recon

    def get_code(self, sess, x):
        z = sess.run(self.z, feed_dict={self.input_x: x})
        return z

    def generate_from_code(self, sess, z):
        x_gen = sess.run(self.decoded_x, feed_dict={self.z: z})
        return x_gen

    def generate(self, sess, num_samples):
        z = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, self.z_dim))
        x_gen = self.generate_from_code(sess, z)
        return x_gen
