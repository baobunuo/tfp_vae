# tfp_vae
Variational Autoencoder using Tensorflow Probability library

![interpolation gif](assets/afb79a4e-2d54-439e-bde0-74e1ebeb6efe.gif)


## Dependencies
```
tensorflow==1.13.1
tensorflow_probability==0.6.0
tensorflow_datasets==1.0.2
numpy==1.16.4
matplotlib==3.1.0
moviepy==1.0.0
```

## Usage

This implementation supports MNIST, CelebA, CIFAR-10, and Omniglot.

To train a VAE on MNIST, you can run:

    $ python app.py --dataset=mnist --img_height=32 --img_width=32 --img_channels=1 --z_dim=100 --mode=train

For a full list of options, including options for checkpointing, tensorboard, and creating visualizations,
you can run:

    $ python app.py --help

And you'll see some options like this:
<details>
  <summary>click to expand view</summary>

```
app.py:
  --activation: <relu|elu>: activation: the activation function for the convolutional layers
    (default: 'relu')
  --batch_size: batch_size: number of examples per minibatch
    (default: '64')
    (an integer)
  --checkpoint_dir: checkpoint_dir: directory for saving model checkpoints
    (default: 'checkpoints/')
  --checkpoint_frequency: checkpoint_frequency: frequency to save checkpoints, measured in global steps
    (default: '500')
    (an integer)
  --dataset: <mnist|celeb_a|cifar10|omniglot>: dataset: which dataset to use
    (default: 'mnist')
  --decoder_res_blocks: decoder_res_blocks: number of blocks in the decoder
    (default: '3')
    (an integer)
  --encoder_res_blocks: encoder_res_blocks: number of blocks in the encoder
    (default: '3')
    (an integer)
  --epochs: epochs: number of epochs to train for. ignored if mode is not 'train'
    (default: '10')
    (an integer)
  --img_channels: img_channels: number of image channels
    (default: '1')
    (an integer)
  --img_height: img_height: height to scale images to, in pixels
    (default: '32')
    (an integer)
  --img_width: img_width: width to scale images to, in pixels
    (default: '32')
    (an integer)
  --load_checkpoint: load_checkpoint: checkpoint directory or checkpoint to load
    (default: '')
  --mode: <train|eval|generate|reconstruct|interpolate|interpolate_gif>: mode: one of train, eval, generate, reconstruct, interpolate, interpolate_gif
    (default: 'train')
  --num_filters: num_filters: number of convolutional filters per layer
    (default: '32')
    (an integer)
  --output_dir: output_dir: directory for visualizations
    (default: 'output/')
  --summaries_dir: summaries_dir: directory for tensorboard logs
    (default: '/tmp/vae_summaries/')
  --z_dim: z_dim: dimension of latent variable z
    (default: '100')
    (an integer)

```
</details>


## Tensorboard
To visualize results using Tensorboard, you can open a second shell, and run:

    $ tensorboard --logdir=/tmp/vae_summaries/ --host=localhost

where `logdir` should be the directory you specified using the `summaries_dir` flag. 

Now you can open a browser, and navigate to `localhost:6006`, and you'll be able to monitor training progress:

![tensorboard](assets/tensorboard.png)