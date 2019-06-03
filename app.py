import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

from tfp_vae.vae import VAE
from tfp_vae.utils.data import get_dataset
import tfp_vae.utils.callbacks as calls
import routines

flags = tf.app.flags


flags.DEFINE_string("mode", 'train', "mode: train, eval, generate, reconstruct, interpolate ['train']")

flags.DEFINE_string("dataset", 'mnist', "dataset: which dataset to use ['mnist']")
flags.DEFINE_integer("img_height", 32, "img_height: height to scale images to, in pixels")
flags.DEFINE_integer("img_width", 32, "img_width: width to scale images to, in pixels")
flags.DEFINE_integer("img_channels", 1, "img_channels: number of image channels")

flags.DEFINE_integer("batch_size", 64, "batch_size: number of examples per minibatch")
flags.DEFINE_integer("z_dim", 100, "z_dim: dimension of latent variable z")

flags.DEFINE_string("summaries_dir", '/tmp/vae_summaries/', "summaries_dir: directory for tensorboard logging")
flags.DEFINE_string("output_dir", 'output/', "output_dir: directory for visualizations")

flags.DEFINE_string("checkpoint_dir", 'checkpoints/', "checkpoint_dir: directory for saving model checkpoints")
flags.DEFINE_string("load_checkpoint", '', "load_checkpoint: checkpoint directory or checkpoint to load")
flags.DEFINE_integer("epochs", 10, "epochs: number of epochs to train for")

FLAGS = flags.FLAGS


def main(_):

    ## hyperparams
    hps = tf.contrib.training.HParams(
        batch_size=FLAGS.batch_size,
        img_height=FLAGS.img_height,
        img_width=FLAGS.img_width,
        img_channels=FLAGS.img_channels,
        discrete_outputs=(True if FLAGS.img_channels == 1 else False),
        z_dim=FLAGS.z_dim,
        activation=tf.nn.elu,
        epochs=FLAGS.epochs
    )

    ## dataset
    ds_train, ds_test = get_dataset(
        name=FLAGS.dataset, hps=hps)

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

    mode_to_routine = {
        'train': routines.train,
        'eval': routines.evaluate,
        'generate': routines.generate,
        #'reconstruct': routines.reconstruct,
        #'interpolate': routines.interpolate
    }
    routine = mode_to_routine[FLAGS.mode]

    if FLAGS.mode == 'train':
        checkpoint_dir = FLAGS.checkpoint_dir
        callbacks = {
            'tensorboard': calls.tensorboard(train_writer), 
            'checkpointing': calls.checkpointing(sess, checkpoint_dir, saver)
        }
        routine(ds_train, sess, model, callbacks)

    elif FLAGS.mode == 'eval':
        callbacks = {}
        routine(ds_test, sess, model, callbacks)

    else:
        output_dir = FLAGS.output_dir
        callbacks = {
            'visualization': calls.visualization(output_dir)
        }
        routine(ds_train, sess, model, callbacks)


if __name__ == '__main__':
    tf.app.run()
