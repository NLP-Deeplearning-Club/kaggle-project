# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from time import gmtime, strftime
from tqdm import tqdm
import glob
import os
import time

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef() # create an empty GraphDef object.
    graph_def.ParseFromString(f.read()) # parse the graph file.
    tf.import_graph_def(graph_def, name='') # import the to the current graph from graph_def.

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)    
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    return labels[predictions.argsort()[-1]].replace("_","")

def main(_):
  """Entry point for script, converts flags to arguments."""
  labels_list = load_labels(FLAGS.labels)
  print(labels_list)
  load_graph(FLAGS.graph)

  folder = "/media/enroutelab/sdd/data/tensorflow_speech_dataset_kaggle/test/audio/"
  print('Writing to submission.csv............')
  submission = dict()
  length = len(glob.glob(folder+"*"))
  #print(length)
  wavlist = sorted(glob.glob(folder+"*"))

  for t in tqdm(range(length)):
    wav = wavlist[t]
    with open(wav, 'rb') as wav_file:
      wav_data = wav_file.read()
    submission[os.path.basename(wav)] = run_graph(wav_data, labels_list, FLAGS.input_name, 
      FLAGS.output_name, FLAGS.how_many_labels)
    print(os.path.basename(wav), submission[os.path.basename(wav)])

  timenow = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
  sub_file = 'submission_' + str(timenow) + '.csv'
  with open(sub_file, 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=1,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
