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

"""Kepler light curve inputs to the AstroWaveNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astrowavenet.data import base


def parse_example(serialized):
  """Parses a single tf.Example proto."""
  features = tf.parse_single_example(
      serialized,
      features={
          "time": tf.VarLenFeature(tf.float32),
          "flux": tf.VarLenFeature(tf.float32),
          "mask": tf.VarLenFeature(tf.int64),
          "kepler_id": tf.FixedLenFeature([], dtype=tf.int64)
      })
  # Extract values from SparseTensor objects.
  time = features["time"].values
  autoregressive_input = features["flux"].values
  mask = tf.cast(features["mask"].values, dtype=tf.float32)
  example_id = tf.cast(features["kepler_id"], dtype=tf.int32)
  return {
      "time": time,
      "autoregressive_input": autoregressive_input,
      "conditioning_stack": mask,
      "example_id": example_id,
      "weights": mask,
  }


class KeplerLightCurves(base.TFRecordDataset):
  """Kepler light curve inputs to the AstroWaveNet model."""

  def create_example_parser(self):
    return parse_example
