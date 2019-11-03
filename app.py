import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

from tfp_vae.vae import VAE
from tfp_vae.utils.data import get_dataset
import tfp_vae.utils.callbacks as calls
import routines

flags = tf.app.flags


flags.DEFINE_enum("mode", 'train', ['train', 'eval', 'generate', 'reconstruct', 'interpolate', 'interpolate_gif'], "mode: one of train, eval, generate, reconstruct, interpolate, interpolate_gif")

flags.DEFINE_enum("dataset", 'mnist', ['mnist', 'celeb_a', 'cifar10', 'omniglot'], "dataset: which dataset to use")
flags.DEFINE_integer("img_height", 32, "img_height: height to scale images to, in pixels")
flags.DEFINE_integer("img_width", 32, "img_width: width to scale images to, in pixels")
flags.DEFINE_integer("img_channels", 1, "img_channels: number of image channels")

flags.DEFINE_integer("batch_size", 64, "batch_size: number of examples per minibatch")
flags.DEFINE_integer("enc_blocks", 3, "enc_blocks: number of blocks in the encoder")
flags.DEFINE_integer("dec_blocks", 3, "dec_blocks: number of blocks in the decoder")
flags.DEFINE_integer("num_filters", 32, "num_filters: number of filters per layer")
flags.DEFINE_integer("code_size", 100, "code_size: dimension of latent variable z")

flags.DEFINE_string("summaries_dir", '/tmp/vae_summaries/', "summaries_dir: directory for tensorboard logs")
flags.DEFINE_string("output_dir", 'output/', "output_dir: directory for visualizations")

flags.DEFINE_string("checkpoint_dir", 'checkpoints/', "checkpoint_dir: directory for saving model checkpoints")
flags.DEFINE_string("load_checkpoint", '', "load_checkpoint: checkpoint directory or checkpoint to load")
flags.DEFINE_integer("checkpoint_frequency", 500, "checkpoint_frequency: frequency to save checkpoints, measured in global steps")

flags.DEFINE_integer("epochs", 10, "epochs: number of epochs to train for. ignored if mode is not 'train'")

FLAGS = flags.FLAGS


def main(_):

    ## hyperparams
    hps = tf.contrib.training.HParams(
        batch_size = FLAGS.batch_size,
        img_height = FLAGS.img_height,
        img_width = FLAGS.img_width,
        img_channels = FLAGS.img_channels,
        num_filters = FLAGS.num_filters,
        code_size = FLAGS.code_size,
        enc_blocks = FLAGS.enc_blocks,
        dec_blocks = FLAGS.dec_blocks,
        discrete_outputs = (True if FLAGS.img_channels == 1 else False),
        epochs = FLAGS.epochs)

    ## dataset
    ds_train, ds_test = get_dataset(name=FLAGS.dataset, hps=hps)

    ## model and session
    model = VAE(hps)
    sess = tf.Session()

    ## tensorboard
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    ## checkpointing
    saver = tf.train.Saver()

    ## init op
    init_op = tf.global_variables_initializer()
    _ = sess.run(init_op)

    ## restoring
    if FLAGS.load_checkpoint != '':
        saver.restore(sess, FLAGS.load_checkpoint)

    ## helper functions for the various modes supported by this application
    mode_to_routine = {
        'train': routines.train,
        'eval': routines.evaluate,
        'generate': routines.generate,
        'reconstruct': routines.reconstruct,
        'interpolate': routines.interpolate,
        'interpolate_gif': routines.interpolate_gif
    }
    routine = mode_to_routine[FLAGS.mode]

    ## rather than pass around tons of arguments,
    #  just use callbacks to perform the required functionality
    if FLAGS.mode == 'train':
        checkpoint_dir = FLAGS.checkpoint_dir
        checkpoint_frequency = FLAGS.checkpoint_frequency
        callbacks = {
            'tensorboard': calls.tensorboard(train_writer), 
            'checkpointing': calls.checkpointing(sess, saver, checkpoint_dir, checkpoint_frequency)
        }
        routine(ds_train, sess, model, callbacks)

    elif FLAGS.mode == 'eval':
        callbacks = {}
        routine(ds_test, sess, model, callbacks)

    else:
        output_dir = FLAGS.output_dir
        callbacks = {
            'save_png': calls.save_png(output_dir),
            'save_gif': calls.save_gif(output_dir)
        }
        routine(ds_train, sess, model, callbacks)


if __name__ == '__main__':
    tf.app.run()
