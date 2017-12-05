import utils
import os
import tensorflow as tf

from time import gmtime, strftime
from tqdm import tqdm
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

print "num_classes:", len(utils.POSSIBLE_LABELS)
print "(name:id):", utils.name2id
trainset, valset = utils.load_data(utils.DATADIR)
print "trainset example:", trainset[0]
print "validset example:", valset[0]
# ------------SUP-parameters--------------------
params = dict(
    seed=2018,
    batch_size=64,
    keep_prob=0.5,
    learning_rate=1e-3,
    clip_gradients=15.0,
    use_batch_norm=True,
    num_classes=len(utils.POSSIBLE_LABELS),
)

hparams = tf.contrib.training.HParams(**params)
if not os.path.exists(utils.MODELDIR):
    os.makedirs(utils.MODELDIR)
run_config = tf.contrib.learn.RunConfig(model_dir=utils.MODELDIR)

train_input_fn = generator_input_fn(
    x=utils.data_generator(trainset, hparams, 'train'),
    # you could leave target_key in features, so labels in model_handler will be empty
    target_key='target',
    batch_size=hparams.batch_size,
    shuffle=True,
    num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10,
    num_threads=1,
)

val_input_fn = generator_input_fn(
    x=utils.data_generator(valset, hparams, 'val'),
    target_key='target',
    batch_size=hparams.batch_size,
    shuffle=True,
    num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10,
    num_threads=1,
)

test_input_fn = generator_input_fn(
    x=utils.test_data_generator(),
    batch_size=hparams.batch_size,
    shuffle=False,
    num_epochs=1,
    queue_capacity=10 * hparams.batch_size,
    num_threads=1,
)

# Let's training!
# ------------SUP-parameters--------------------


def _create_my_experiment(run_config, hparams):
    exp = tf.contrib.learn.Experiment(
        estimator=tf.estimator.Estimator(
            model_fn=utils.model_handler, config=run_config, params=hparams),
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        train_steps=10000,  # just randomly selected params
        eval_steps=200,  # read source code for steps-epochs ariphmetics
        train_steps_per_iteration=1000,
    )
    return exp


print('Start training............')
tf.contrib.learn.learn_runner.run(experiment_fn=_create_my_experiment,
                                  run_config=run_config,
                                  schedule="continuous_train_and_eval",
                                  hparams=hparams)

# Let's predicting!
print('Start predicting............')
model = tf.estimator.Estimator(
    model_fn=utils.model_handler, config=run_config, params=hparams)
predict = model.predict(input_fn=test_input_fn)

# last batch will contain padding, so remove duplicates
print('Writing to submission.csv............')
submission = dict()
for t in tqdm(predict):
    fname, label = t['sample'].decode(), utils.id2name[t['label']]
    submission[fname] = label

timenow = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
sub_file = 'submission_' + str(timenow) + '.csv'
with open(sub_file, 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))
