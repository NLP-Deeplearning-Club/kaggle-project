import os
import re
from glob import glob
import numpy as np
from scipy.io import wavfile

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import signal

DATADIR = './dataset'  # unzipped train and test data
MODELDIR = './model'  # just a random name
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}


def load_data(data_dir):
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val


def data_generator(data, params, mode='train'):
    def generator():
        if mode == 'train':
            np.random.shuffle(data)
        # ------------ADD Augmentation HERE-------------

        # ----------------------------------------------
        for (label_id, uid, fname) in data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:  # if > L, choose randomly the start point
                        begin = np.random.randint(0, len(wav) - L)
                    else:
                        begin = 0
                    yield dict(target=np.int32(label_id), wav=wav[begin: begin + L])
            except Exception as err:
                print(err, label_id, uid, fname)

    return generator


def test_data_generator():
    def generator():
        for path in glob(os.path.join(DATADIR, 'test/audio/*wav')):
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            fname = os.path.basename(path)
            yield dict(sample=np.string_(fname), wav=wav)
    return generator

# ------------NEURALNET STRUCTURE--------------------


def baseline(x, params, is_training):
    x = layers.batch_norm(x, is_training=is_training)
    for i in range(4):
        x = layers.conv2d(
            x, 16 * (2 ** i), 3, 1,
            activation_fn=tf.nn.elu,
            normalizer_fn=layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
        )
        x = layers.max_pool2d(x, 2, 2)
    # just take two kind of pooling and then mix them, why not :)
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    x = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)
    # again conv2d 1x1 instead of dense layer
    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)
    return tf.squeeze(logits, [1, 2])


def model_handler(features, labels, mode, params, config):
    # I'm really like to use make_template instead of variable_scopes and re-usage
    extractor = tf.make_template('extractor', baseline, create_scope_now_=True)
    # wav is a waveform signal with shape (16000, )
    wav = features['wav']
    # we want to compute spectograms by means of short time fourier transform:
    specgram = signal.stft(
        wav,
        # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        400,
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into phase and amp parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))

    x = tf.stack([amp, phase], axis=3)  # shape is [bs, time, freq_bins, 2]
    x = tf.to_float(x)  # we want to have float32, not float64

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
        # some lr tuner, you could use move interesting functions

        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(
                lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        specs = dict(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
        specs = dict(mode=mode, loss=loss,
                     eval_metric_ops=dict(acc=(acc, acc_op)))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # for probability just take tf.nn.softmax()
            'label': tf.argmax(logits, axis=-1),
            'sample': features['sample'],  # it's a hack for simplicity
        }
        specs = dict(mode=mode, predictions=predictions)

    return tf.estimator.EstimatorSpec(**specs)
