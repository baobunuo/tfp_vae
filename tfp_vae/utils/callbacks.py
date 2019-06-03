import uuid
import os
import matplotlib.pyplot as plt
import numpy as np


def checkpointing(sess, checkpoint_dir, saver):
    func = lambda: saver.save(sess, checkpoint_dir)
    return func

def tensorboard(writer):
    def func(step, summary):
        writer.add_summary(summary, step)
        writer.flush()

    return func

def visualization(output_dir):
    def func(img):
        filename = str(uuid.uuid4()) + '.png'
        fp = os.path.join(output_dir, filename)
        if img.shape[-1] == 1:
            img = np.concatenate([img for _ in range(0,3)], axis=-1)
        plt.imsave(fp, img)
        return fp

    return func
