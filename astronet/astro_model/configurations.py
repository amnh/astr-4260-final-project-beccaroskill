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

"""Configurations for model building, training and evaluation.

The default base configuration has one "global_view" time series feature per
input example. Additional time series features and auxiliary features can be
added.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def base():
  """Returns the base config for model building, training and evaluation."""
  return {
      # Configuration for reading input features and labels.
      "inputs": {
          # Feature specifications.
          "features": {
              "global_view": {
                  "length": 2001,
                  "is_time_series": True,
                  "subcomponents": [],
              },
          },

          # Name of the feature containing training labels.
          "label_feature": "av_training_set",

          # Label string to integer id. Use -1 to ignore examples with that
          # label.
          "label_map": {
              "PC": 1,  # Planet Candidate.
              "AFP": 0,  # Astrophysical False Positive.
              "NTP": 0,  # Non-Transiting Phenomenon.
              "SCR1": 0,  # TCE from scrambled light curve with SCR1 order.
              "INV": 0,  # TCE from inverted light curve.
              "INJ1": 1,  # Injected Planet.
              "INJ2": 0,  # Simulated eclipsing binary.
          },
      },
      # Hyperparameters for building and training the model.
      "hparams": {
          # Number of output dimensions (predictions) for the classification
          # task. If >= 2 then a softmax output layer is used. If equal to 1
          # then a sigmoid output layer is used.
          "output_dim": 1,

          # Fully connected layers before the logits layer.
          "num_pre_logits_hidden_layers": 0,
          "pre_logits_hidden_layer_size": 0,
          "pre_logits_dropout_rate": 0.0,

          # Number of examples per training batch.
          "batch_size": 256,

          # Learning rate parameters.
          "learning_rate": 2e-4,
          "learning_rate_decay_steps": 0,
          "learning_rate_end_factor": 0.0,
          "learning_rate_decay_power": 1.0,

          # Weight decay regularization.
          "weight_decay": 0.0,

          # Label smoothing regularization.
          "label_smoothing": 0.0,

          # Optimizer for training the model.
          "optimizer": "adam",

          # If not None, gradient norms will be clipped to this value.
          "clip_gradient_norm": None,
      }
  }
