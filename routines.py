import tensorflow_datasets as tfds


def train(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):
        elbo, step, summary = model.train(sess, batch)
        print('Global Step {}, ELBO {}'.format(step, elbo))

        _ = callbacks['tensorboard'](step, summary)

    _ = callbacks['checkpointing']()


def evaluate(dataset, sess, model, callbacks):
    elbo_sum = 0.0
    batch_counter = 0

    for batch in tfds.as_numpy(dataset):

        elbo = model.eval(sess, batch)
        elbo_sum += elbo
        batch_count += 1

    elbo_avg = elbo_sum / float(batch_count)
    print('Test ELBO: {}'.format(elbo_avg))


def generate(dataset, sess, model, callbacks):
    gen_xs = model.generate(sess, num_samples=64)

    print('saving images...')
    for x in gen_xs:
        fp = callbacks['visualization'](x)
        print(fp)
