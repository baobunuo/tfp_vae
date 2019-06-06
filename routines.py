import tensorflow_datasets as tfds
import numpy as np


def train(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):
        elbo, step, summary = model.train(sess, batch)
        print('Global Step {}, ELBO {}'.format(step, elbo))

        _ = callbacks['tensorboard'](step, summary)
        _ = callbacks['checkpointing'](step)


def evaluate(dataset, sess, model, callbacks):
    elbo_sum = 0.0
    batch_count = 0

    for batch in tfds.as_numpy(dataset):

        elbo = model.evaluate(sess, batch)
        elbo_sum += elbo
        batch_count += 1

    elbo_avg = elbo_sum / float(batch_count)
    print('Test ELBO: {}'.format(elbo_avg))


def generate(dataset, sess, model, callbacks):
    gen_xs = model.generate(sess, num_samples=64)

    print('saving images...')
    rows = []
    for i in range(0, 8):
        row = []
        for j in range(0, 8):
            row.append(gen_xs[8 * i + j])
        row = np.concatenate(row, axis=1)
        rows.append(row)

    img = np.concatenate(rows, axis=0)
    fp = callbacks['save_png'](img)
    print(fp)


def reconstruct(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):
        batch_size = batch.shape[0]
        xs = batch
        recon_xs = model.reconstruct(sess, batch)

        print('saving images...')
        columns = []
        for i in range(0, batch_size):
            col = []
            col.append(xs[i])
            col.append(recon_xs[i])
            col = np.concatenate(col, axis=0)
            columns.append(col)
        img = np.concatenate(columns, axis=1)
        fp = callbacks['save_png'](img)
        print(fp)

        break

def interpolate(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):
        batch_size = batch.shape[0]
        xs = batch
        permutation = np.random.permutation(batch_size)
        permuted_xs = xs[permutation]
        z0 = model.get_code(sess, xs)
        zT = model.get_code(sess, permuted_xs)
        n = 5
        ts = [(t / float(n - 1)) for t in range(0, n)]
        zs = [(1. - t) * z0 + t * zT for t in ts]

        print('saving images...')
        columns = []
        col = np.concatenate(xs, axis=0)
        columns.append(col)
        for i in range(0, n):
            zs_i = zs[i]
            xs_i = model.generate_from_code(sess, zs_i)
            col = np.concatenate(xs_i, axis=0)
            columns.append(col)
        col = np.concatenate(permuted_xs, axis=0)
        columns.append(col)

        img = np.concatenate(columns, axis=1)
        fp = callbacks['save_png'](img)
        print(fp)

        break


def interpolation_gif(dataset, sess, model, callbacks):

    for batch in tfds.as_numpy(dataset):
        batch_size = batch.shape[0]
        xs = batch
        permutation = np.random.permutation(batch_size)
        permuted_xs = xs[permutation]
        z0 = model.get_code(sess, xs)
        zT = model.get_code(sess, permuted_xs)
        n = 8
        ts = [(t / float(n - 1)) for t in range(0, n)]
        zs = [(1. - t) * z0 + t * zT for t in ts]
        Z = np.stack(zs, axis=0)  # [T, B, Z]
        Z = np.transpose(Z, [1, 0, 2])

        print('saving images...')
        for b in range(0, batch_size):
            img_frames = []
            img_frames.append(xs[b])
            for i in range(0, n):
                z_i = Z[b][i]
                z_i = np.expand_dims(z_i, 0)
                x_i = model.generate_from_code(sess, z_i)
                x_i = np.squeeze(x_i, 0)
                img_frames.append(x_i)
            img_frames.append(permuted_xs[b])
            
            if x_i.shape[-1] == 1:
                transform = lambda x_i: np.concatenate([x_i, x_i, x_i], axis=-1)
                img_frames = list(map(transform, img_frames))
            elif x_i.shape[-1] == 3:
                transform = lambda x_i: 0.5 * x_i + 0.5
                img_frames = list(map(transform, img_frames))

            print(len(img_frames))
            print(img_frames[0].shape)
            fp = callbacks['save_gif'](img_frames)
            print(fp)

        break
