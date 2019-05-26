import tensorflow as tf
import numpy as np

class VAE:
    def __init__(self, hps):
        assert hps.img_height % 8 == 0 and hps.img_width % 8 == 0

        self.img_height = hps.img_height
        self.img_width = hps.img_width
        self.img_channels = hps.img_channels
        self.z_dim = hps.z_dim
        self.num_filters = 32
        self.activation = tf.nn.relu
        self.discrete_outputs = hps.discrete_outputs
        self.global_step = tf.train.get_or_create_global_step()

        self._name = 'VAE'

        with tf.variable_scope(self._name):
            self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            qz_given_x = self.qz_given_x(self.input_x)
            self.z = qz_given_x.sample()

            self.mu = qz_given_x.mean()
            self.sigma = qz_given_x.stddev()
            self.kl_div = 0.5 * tf.reduce_sum((tf.square(self.mu) + tf.square(self.sigma) - 2.0 * tf.log(self.sigma + 1e-6) - 1.0), axis=1)
            
            px_given_z = self.px_given_z(self.z)

            self.log_px_given_z = tf.reduce_sum(px_given_z.log_prob(self.input_x), axis=[1, 2, 3])

            self.elbo_x = self.log_px_given_z - self.kl_div

            self.elbo = tf.reduce_mean(self.elbo_x, axis=0)
            self.loss = -self.elbo

            self.optimizer = tf.train.AdamOptimizer()
            tvars = [v for v in tf.trainable_variables() if v.name.startswith(self._name)]
            self.gradients, _ = zip(*self.optimizer.compute_gradients(self.loss, tvars))
            self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 10.0)
            self.train_op = self.optimizer.apply_gradients(
                grads_and_vars=zip(self.clipped_gradients, tvars),
                global_step=self.global_step)

            self.decoded_x = px_given_z.mean()

    def train(self, sess, x):
        _, elbo = sess.run([self.train_op, self.elbo], feed_dict={self.input_x: x})
        return elbo

    def evaluate(self, sess, x):
        elbo = sess.run(self.elbo, feed_dict={self.input_x: x})
        return elbo

    def reconstruct(self, sess, x):
        x_recon = sess.run(self.decoded_x, feed_dict={self.input_x: x})
        return x_recon

    def get_code(self, sess, x):
        z = sess.run(self.mu, feed_dict={self.input_x: x})
        return z

    def generate_from_code(self, sess, z):
        x_gen = sess.run(self.decoded_x, feed_dict={self.z: z})
        return x_gen

    def generate(self, sess, num_samples):
        z = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, self.z_dim))
        x_gen = self.generate_from_code(sess, z)
        return x_gen

    def interpolate(self, sess, x1, x2):
        xs = np.stack([x1, x2], axis=0)
        zs = self.get_code(sess, xs)
        z1 = zs[0]
        z2 = zs[1]

        pts = 5
        ts = [(float(i) / float(pts-1)) for i in range(0, pts)]
        interpolation_zs = [(1.0 - t) * z1 + t * z2 for t in ts]
        x_gen = self.generate_from_code(sess, interpolation_zs)
        return x_gen

    def pz(self, batch_size):
        mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.z_dim])
        logsigma = tf.zeros(dtype=tf.float32, shape=[batch_size, self.z_dim])
        z_dist = tf.distributions.Normal(loc=mu, scale=tf.exp(logsigma))
        return z_dist
            
    def qz_given_x(self, x):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            r1 = self.encoder_res_block(x, name='block1')
            r2 = self.encoder_res_block(r1, name='block2')
            r3 = self.encoder_res_block(r2, name='block3')
            flat = tf.layers.flatten(r3)
            enc_x = tf.layers.dense(flat, units=(2 * self.z_dim), activation=None)
            mu = tf.layers.dense(enc_x, units=self.z_dim, activation=None)
            logsigma = tf.layers.dense(enc_x, units=self.z_dim, activation=None)
            z_dist = tf.distributions.Normal(loc=mu, scale=tf.exp(logsigma))
            return z_dist

    def px_given_z(self, z):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            scaling = (2 ** 3)

            decoder_input_fc1 = tf.layers.dense(
                z, units=((self.img_height // scaling) * (self.img_width // scaling) * self.num_filters),
                activation=None)

            decoder_input_3d = tf.reshape(
                decoder_input_fc1, shape=[-1, (self.img_height // scaling), (self.img_width // scaling), self.num_filters])

            r1 = self.decoder_res_block(decoder_input_3d, name='block1')
            r2 = self.decoder_res_block(r1, name='block2')
            r3 = self.decoder_res_block(r2, name='block3')
            res_tower_output = r3

            if self.discrete_outputs:
                decoded_px = tf.layers.conv2d_transpose(
                    res_tower_output, filters=self.img_channels, kernel_size=1, strides=1, padding='same',
                    activation=tf.nn.sigmoid)
                x_dist = tf.distributions.Bernoulli(probs=decoded_px)
                return x_dist
            else:
                decoded_mu_x = tf.layers.conv2d_transpose(
                    res_tower_output, filters=self.img_channels, kernel_size=1, strides=1, padding='same',
                    activation=tf.nn.tanh)

                decoded_var_x = 0.01 + 1.0 + tf.layers.conv2d_transpose(
                    res_tower_output, filters=self.img_channels, kernel_size=1, strides=1, padding='same',
                    activation=tf.nn.elu)

                x_dist = tf.distributions.Normal(loc=decoded_mu_x, scale=tf.sqrt(decoded_var_x))
                return x_dist

    def encoder_res_block(self, inputs, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            c0 = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='valid',
                activation=self.activation)

            c1 = tf.layers.conv2d(
                c0, filters=self.num_filters, kernel_size=4, strides=1, padding='same',
                activation=self.activation)

            c2 = tf.layers.conv2d(
                c1, filters=self.num_filters, kernel_size=4, strides=1, padding='same',
                activation=self.activation)

            return c0 + c2

    def decoder_res_block(self, inputs, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            d0 = tf.layers.conv2d_transpose(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='same',
                activation=self.activation)

            d1 = tf.layers.conv2d_transpose(
                d0, filters=self.num_filters, kernel_size=4, strides=1, padding='same',
                activation=self.activation)

            d2 = tf.layers.conv2d_transpose(
                d1, filters=self.num_filters, kernel_size=4, strides=1, padding='same',
                activation=self.activation)

            return d0 + d2
