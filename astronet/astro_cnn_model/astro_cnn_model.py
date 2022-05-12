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

"""A model for classifying light curves using a convolutional neural network.

See the base class (in astro_model.py) for a description of the general
framework of AstroModel and its subclasses.

The architecture of this model is:


                                     predictions
                                          ^
                                          |
                                       logits
                                          ^
                                          |
                                (fully connected layers)
                                          ^
                                          |
                                   pre_logits_concat
                                          ^
                                          |
                                    (concatenate)

              ^                           ^                          ^
              |                           |                          |
   (convolutional blocks 1)  (convolutional blocks 2)   ...          |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1     time_series_feature_2    ...     aux_features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.astro_model import astro_model


class AstroCNNModel(astro_model.AstroModel):
  """A model for classifying light curves using a convolutional neural net."""

  def _build_cnn_layers(self, inputs, hparams, scope="cnn"):
    """Builds convolutional layers.

    The layers are defined by convolutional blocks with pooling between blocks
    (but not within blocks). Within a block, all layers have the same number of
    filters, which is a constant multiple of the number of filters in the
    previous block. The kernel size is fixed throughout.

    Args:
      inputs: A Tensor of shape [batch_size, length] or
        [batch_size, length, ndims].
      hparams: Object containing CNN hyperparameters.
      scope: Prefix for operation names.

    Returns:
      A Tensor of shape [batch_size, output_size], where the output size depends
      on the input size, kernel size, number of filters, number of layers,
      convolution padding type and pooling.
    """
    with tf.name_scope(scope):
      net = inputs
      if net.shape.rank == 2:
        net = tf.expand_dims(net, -1)  # [batch, length] -> [batch, length, 1]
      if net.shape.rank != 3:
        raise ValueError(
            "Expected inputs to have rank 2 or 3. Got: {}".format(inputs))
      for i in range(hparams.cnn_num_blocks):
        num_filters = int(hparams.cnn_initial_num_filters *
                          hparams.cnn_block_filter_factor**i)
        with tf.name_scope("block_{}".format(i + 1)):
          for j in range(hparams.cnn_block_size):
            conv_op = tf.keras.layers.Conv1D(
                filters=num_filters,
                kernel_size=int(hparams.cnn_kernel_size),
                padding=hparams.convolution_padding,
                activation=tf.nn.relu,
                name="conv_{}".format(j + 1))
            print(conv_op, num_filters)
            net = conv_op(net)

          if hparams.pool_size > 1:  # pool_size 0 or 1 denotes no pooling
            pool_op = tf.keras.layers.MaxPool1D(
                pool_size=int(hparams.pool_size),
                strides=int(hparams.pool_strides),
                name="pool")
            print(pool_op, int(hparams.pool_size))
            net = pool_op(net)

      # Flatten.
      # print('Model', net.summary())
      net.shape.assert_has_rank(3)
      net_shape = net.shape.as_list()
      output_dim = net_shape[1] * net_shape[2]
      net = tf.reshape(net, [-1, output_dim], name="flatten")

    return net

  def build_time_series_hidden_layers(self):
    """Builds hidden layers for the time series features.

    Inputs:
      self.time_series_features

    Outputs:
      self.time_series_hidden_layers
    """
    time_series_hidden_layers = {}
    for name, time_series in self.time_series_features.items():
      time_series_hidden_layers[name] = self._build_cnn_layers(
          inputs=time_series,
          hparams=self.hparams.time_series_hidden[name],
          scope=name + "_hidden")

    self.time_series_hidden_layers = time_series_hidden_layers
