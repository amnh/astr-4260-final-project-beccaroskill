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

"""A TensorFlow WaveNet model for generative modeling of light curves.

Implementation based on "WaveNet: A Generative Model of Raw Audio":
https://arxiv.org/abs/1609.03499
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


def _shift_right(x, n):
  """Shifts the input Tensor right by n indices along the second dimension.

  Pads the front of the sequence with zeros and discards the last n elements.

  Args:
    x: Input three-dimensional tf.Tensor.
    n: Integer; number of indices to shift.

  Returns:
    Padded, shifted tensor of same shape as input.
  """
  x_padded = tf.pad(x, [[0, 0], [n, 0], [0, 0]])
  return x_padded[:, :-n, :]


class AstroWaveNet(object):
  """A TensorFlow model for generative modeling of light curves."""

  def __init__(self, features, hparams, mode):
    """Basic setup.

    The actual TensorFlow graph is constructed in build().

    Args:
      features: A dictionary containing "autoregressive_input" and
        "conditioning_stack", each of which is a named input Tensor. All
        features have dtype float32 and shape [batch_size, length, dim].
      hparams: A ConfigDict of hyperparameters for building the model.
      mode: A tf.estimator.ModeKeys to specify whether the graph should be built
        for training, evaluation or prediction.

    Raises:
      ValueError: If mode is invalid.
    """
    valid_modes = [
        tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
        tf.estimator.ModeKeys.PREDICT
    ]
    if mode not in valid_modes:
      raise ValueError("Expected mode in {}. Got: {}".format(valid_modes, mode))
    self.features = features
    self.hparams = hparams
    self.mode = mode

    # Input Tensors.
    self.autoregressive_input = None
    self.conditioning_stack = None
    self.weights = None

    # Model Tensors.
    self.network_output = None  # Sum of skip connections from dilation stack.
    self.dist_params = None  # Dict of predicted distribution parameters.
    self.predicted_distributions = None  # Predicted distribution for examples.
    self.autoregressive_target = None  # Autoregressive target predictions.
    self.batch_losses = None  # Loss for each predicted distribution in batch.
    self.per_example_loss = None  # Loss for each example in batch.
    self.num_nonzero_weight_examples = None  # Number of examples in batch.
    self.total_loss = None  # Overall loss for the batch.
    self.global_step = None  # Global step Tensor.

  def build_inputs(self):
    """Builds the input Tensors.

    Inputs:
      self.features

    Outputs:
      self.autoregressive_input
      self.conditioning_stack
      self.weights
    """
    self.autoregressive_input = self.features["autoregressive_input"]
    self.conditioning_stack = self.features["conditioning_stack"]

    weights = self.features.get("weights")
    if weights is None:
      weights = tf.ones_like(self.autoregressive_input)
    self.weights = weights

  def causal_conv_layer(self, x, output_size, kernel_width, dilation_rate=1):
    """Applies a dilated causal convolution to the input.

    Args:
      x: tf.Tensor; Input tensor.
      output_size: int; Number of output filters for the convolution.
      kernel_width: int; Width of the 1D convolution window.
      dilation_rate: int; Dilation rate of the layer.

    Returns:
      Resulting tf.Tensor after applying the convolution.
    """
    causal_conv_op = tf.keras.layers.Conv1D(
        output_size,
        kernel_width,
        padding="causal",
        dilation_rate=dilation_rate,
        name="causal_conv")
    return causal_conv_op(x)

  def conv_1x1_layer(self, x, output_size, activation=None):
    """Applies a 1x1 convolution to the input.

    Args:
      x: tf.Tensor; Input tensor.
      output_size: int; Number of output filters for the 1x1 convolution.
      activation: Activation function to apply (e.g. 'relu').

    Returns:
      Resulting tf.Tensor after applying the 1x1 convolution.
    """
    conv_1x1_op = tf.keras.layers.Conv1D(
        output_size, 1, activation=activation, name="conv1x1")
    return conv_1x1_op(x)

  def gated_residual_layer(self, x, conditioning_stack, dilation_rate):
    """Creates a gated, dilated convolutional layer with a residual connection.

    Args:
      x: tf.Tensor; Input tensor.
      conditioning_stack: tf.Tensor; The conditioning stack corresponding to x.
      dilation_rate: int; Dilation rate of the layer.

    Returns:
      skip_connection: tf.Tensor; Skip connection to network_output layer.
      residual_connection: tf.Tensor; Sum of learned residual and input tensor.
    """
    with tf.name_scope("filter"):
      x_filter_conv = self.causal_conv_layer(x, x.shape[-1].value,
                                             self.hparams.dilation_kernel_width,
                                             dilation_rate)
      cond_filter_conv = self.conv_1x1_layer(conditioning_stack,
                                             x.shape[-1].value)
    with tf.name_scope("gate"):
      x_gate_conv = self.causal_conv_layer(x, x.shape[-1].value,
                                           self.hparams.dilation_kernel_width,
                                           dilation_rate)
      cond_gate_conv = self.conv_1x1_layer(conditioning_stack,
                                           x.shape[-1].value)

    gated_activation = (
        tf.tanh(x_filter_conv + cond_filter_conv) *
        tf.sigmoid(x_gate_conv + cond_gate_conv))

    with tf.name_scope("residual"):
      residual = self.conv_1x1_layer(gated_activation, x.shape[-1].value)
    with tf.name_scope("skip"):
      skip_connection = self.conv_1x1_layer(gated_activation,
                                            self.hparams.skip_output_dim)

    return skip_connection, x + residual

  def build_network(self):
    """Builds WaveNet network.

    This consists of:
      1) An initial causal convolution,
      2) The dilation stack, and
      3) Summing of skip connections

    The network output can then be used to predict various output distributions.

    Inputs:
      self.autoregressive_input
      self.conditioning_stack

    Outputs:
      self.network_output; tf.Tensor
    """
    x = self.autoregressive_input
    conditioning_stack = self.conditioning_stack

    # Shift the input sequence N points to the right (i.e. pad the beginning
    # of the sequence with zeros and drop the last N points) so that the i-th
    # element of the resulting sequence (used as input to the network) aligns
    # with the (i+N)-th element of the original sequence (used as the training
    # target). If self.hparams.use_future_context is True, the conditioning
    # stack is not shifted, so the conditioning stack up to index i+N is used to
    # predict the (i+N)-th element of the input sequence. Otherwise, the
    # conditioning stack is also shifted, so the conditioning stack up to index
    # i is used to predict the (i+N)-th element of the original sequence. No
    # shifting is applied in PREDICT mode, which is used to generate embeddings.
    if self.mode != tf.estimator.ModeKeys.PREDICT:
      shift_num_steps = self.hparams.predict_n_steps_ahead
      x = _shift_right(x, shift_num_steps)
      if not self.hparams.use_future_context:
        conditioning_stack = _shift_right(conditioning_stack, shift_num_steps)

    skip_connections = []
    with tf.name_scope("preprocess"):
      x = self.causal_conv_layer(x, self.hparams.preprocess_output_size,
                                 self.hparams.preprocess_kernel_width)
    for i in range(self.hparams.num_residual_blocks):
      with tf.name_scope("block_{}".format(i)):
        for dilation_rate in self.hparams.dilation_rates:
          with tf.name_scope("dilation_{}".format(dilation_rate)):
            skip_connection, x = self.gated_residual_layer(
                x, conditioning_stack, dilation_rate)
            skip_connections.append(skip_connection)

    self.network_output = tf.add_n(skip_connections)

  def dist_params_layer(self, x, outputs_size):
    """Converts x to the correct shape for populating a distribution object.

    Args:
      x: A Tensor of shape [batch_size, time_series_length, num_features].
      outputs_size: The number of parameters needed to specify all the
        distributions in the output. E.g. 5*3=15 to specify 5 distributions with
        3 parameters each.

    Returns:
      The parameters of each distribution, a tensor of shape [batch_size,
        time_series_length, outputs_size].
    """
    with tf.name_scope("dist_params"):
      conv_outputs = self.conv_1x1_layer(x, outputs_size)
    return conv_outputs

  def build_predictions(self):
    """Predicts output distribution from network outputs.

    Runs the model through:
      1) ReLU
      2) 1x1 convolution
      3) ReLU
      4) 1x1 convolution

    The result of the last convolution is used as the parameters of the
    specified output distribution (currently either Categorical or Normal).

    Inputs:
      self.network_outputs

    Outputs:
      self.dist_params
      self.predicted_distributions

    Raises:
      ValueError: If distribution type is neither 'categorical' nor 'normal'.
    """
    with tf.name_scope("postprocess"):
      network_output = tf.keras.activations.relu(self.network_output)
      network_output = self.conv_1x1_layer(
          network_output,
          output_size=network_output.shape[-1].value,
          activation="relu")
    num_dists = self.autoregressive_input.shape[-1].value

    if self.hparams.output_distribution.type == "categorical":
      num_classes = self.hparams.output_distribution.num_classes
      logits = self.dist_params_layer(network_output, num_dists * num_classes)
      logits_shape = tf.concat(
          [tf.shape(network_output)[:-1], [num_dists, num_classes]], 0)
      logits = tf.reshape(logits, logits_shape)
      dist = tfp.distributions.Categorical(logits=logits)
      dist_params = {"logits": logits}
    elif self.hparams.output_distribution.type == "normal":
      loc_scale = self.dist_params_layer(network_output, num_dists * 2)
      loc, scale = tf.split(loc_scale, 2, axis=-1)
      # Ensure scale is positive.
      scale = tf.nn.softplus(scale) + self.hparams.output_distribution.min_scale
      # Give loc and scale explicit names in the graph.
      loc = tf.identity(loc, "loc")
      scale = tf.identity(scale, "scale")
      dist_params = {"loc": loc, "scale": scale}
      predict_outlier_distribution = self.hparams.output_distribution.get(
          "predict_outlier_distribution", False)
      if predict_outlier_distribution:
        # Create scalar variables representing the probability of each point
        # being an outlier, the mean of the outlier Gaussian distribution, and
        # the standard deviation.
        for name, initial_value in [("outlier_prob", 0.5), ("outlier_loc", 0),
                                    ("outlier_scale", 1)]:
          var = tf.keras.backend.variable(
              [initial_value] * num_dists, dtype=tf.float32, name=name)
          # Wrapping in a tf.identity allows values to be fed in unit tests.
          dist_params[name] = tf.identity(var)

        # Replicate the outlier probability, mean, and standard deviation across
        # all points in all light curves.
        mask = tf.ones_like(loc)
        outlier_prob = mask * dist_params["outlier_prob"]
        outlier_loc = mask * dist_params["outlier_loc"]
        outlier_scale = mask * dist_params["outlier_scale"]

        # Create the categorical probabilities and the mean and standard
        # deviation of the Gaussian mixture.
        mixture_probs = tf.stack([1 - outlier_prob, outlier_prob], axis=-1)
        mixture_loc = tf.stack([loc, outlier_loc], axis=-1)
        mixture_scale = tf.stack([scale, outlier_scale], axis=-1)

        dist = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=mixture_probs),
            components_distribution=tfp.distributions.Normal(
                mixture_loc, mixture_scale))
      else:
        dist = tfp.distributions.Normal(loc, scale)
    else:
      raise ValueError("Unsupported distribution type {}".format(
          self.hparams.output_distribution.type))

    self.dist_params = dist_params
    self.predicted_distributions = dist

  def build_losses(self):
    """Builds the training losses.

    Inputs:
      self.predicted_distributions

    Outputs:
      self.batch_losses
      self.total_loss
    """
    autoregressive_target = self.autoregressive_input

    # Quantize the target if the output distribution is categorical.
    if self.hparams.output_distribution.type == "categorical":
      min_val = self.hparams.output_distribution.min_quantization_value
      max_val = self.hparams.output_distribution.max_quantization_value
      num_classes = self.hparams.output_distribution.num_classes
      clipped_target = tf.keras.backend.clip(autoregressive_target, min_val,
                                             max_val)
      quantized_target = tf.floor(
          (clipped_target - min_val) / (max_val - min_val) * num_classes)
      # Deal with the corner case where clipped_target equals max_val by mapping
      # the label num_classes to num_classes - 1. Essentially, this makes the
      # final quantized bucket a closed interval while all the other quantized
      # buckets are half-open intervals.
      quantized_target = tf.where(
          quantized_target >= num_classes,
          tf.ones_like(quantized_target) * (num_classes - 1), quantized_target)
      autoregressive_target = quantized_target

    autoregressive_target = tf.identity(autoregressive_target, "target")
    log_prob = self.predicted_distributions.log_prob(autoregressive_target)

    weights_dim = len(self.weights.shape)
    per_example_weight = tf.reduce_sum(
        self.weights, axis=list(range(1, weights_dim)))
    per_example_indicator = tf.cast(per_example_weight > 0, tf.float32)
    num_examples = tf.reduce_sum(per_example_indicator)

    batch_losses = -log_prob * self.weights
    losses_ndims = batch_losses.shape.ndims
    per_example_loss_sum = tf.reduce_sum(
        batch_losses, axis=list(range(1, losses_ndims)))
    per_example_loss = tf.where(per_example_weight > 0,
                                per_example_loss_sum / per_example_weight,
                                tf.zeros_like(per_example_weight))
    total_loss = tf.reduce_sum(per_example_loss) / num_examples

    self.autoregressive_target = autoregressive_target
    self.batch_losses = batch_losses
    self.per_example_loss = per_example_loss
    self.num_nonzero_weight_examples = num_examples
    self.total_loss = total_loss

  def build(self):
    """Creates all ops for training, evaluation or inference."""
    self.global_step = tf.train.get_or_create_global_step()
    self.build_inputs()
    self.build_network()
    self.build_predictions()
    self.build_losses()
