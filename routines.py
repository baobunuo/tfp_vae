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
    fp = callbacks['visualization'](img)
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
        fp = callbacks['visualization'](img)
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
        fp = callbacks['visualization'](img)
        print(fp)

        break
