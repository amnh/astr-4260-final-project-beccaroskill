# Copyright 2018 The Exoplanet ML Authors.
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

"""Script for training an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet import models
from astronet.util import estimator_util
from tf_util import config_util
from tf_util import configdict
from tf_util import estimator_runner

tfrecord_dir = "FinalProject/Data/tfrecord"
model = "AstroCNNModel"
config_name = "local_global"
config_json = None
train_files=f"{tfrecord_dir}/train*"
eval_files=f"{tfrecord_dir}/val*"
model_dir = "FinalProject/Data/bogus"
shuffle_buffer_size = 15000
train_steps = 625


def main(_):
  model_class = models.get_model_class(model)

  # Look up the model configuration.
  assert (config_name is None) != (config_json is None), (
      "Exactly one of --config_name or --config_json is required.")
  config = (
      models.get_model_config(model, config_name)
      if config_name else config_util.parse_json(config_json))

  config = configdict.ConfigDict(config)
  config_util.log_and_save_config(config, model_dir)

  # Create the estimator.
  run_config = tf.estimator.RunConfig(keep_checkpoint_max=1)
  estimator = estimator_util.create_estimator(model_class, config.hparams,
                                              run_config, model_dir)

  # Create an input function that reads the training dataset. We iterate through
  # the dataset once at a time if we are alternating with evaluation, otherwise
  # we iterate infinitely.
  train_input_fn = estimator_util.create_input_fn(
      file_pattern=train_files,
      input_config=config.inputs,
      mode=tf.estimator.ModeKeys.TRAIN,
      shuffle_values_buffer=shuffle_buffer_size,
      repeat=1 if eval_files else None)

  if not eval_files:
    estimator.train(train_input_fn, max_steps=train_steps)
  else:
    eval_input_fn = estimator_util.create_input_fn(
        file_pattern=eval_files,
        input_config=config.inputs,
        mode=tf.estimator.ModeKeys.EVAL)
    eval_args = [{"name": "val", "input_fn": eval_input_fn}]

    for _ in estimator_runner.continuous_train_and_eval(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_args=eval_args,
        train_steps=train_steps):
      # continuous_train_and_eval() yields evaluation metrics after each
      # training epoch. We don't do anything here.
      pass


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
