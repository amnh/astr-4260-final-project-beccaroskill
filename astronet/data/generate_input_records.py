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

r"""Script to preprocesses data from the Kepler space telescope.

This script produces training, validation and test sets of labeled Kepler
Threshold Crossing Events (TCEs). A TCE is a detected periodic event on a
particular Kepler target star that may or may not be a transiting planet. Each
TCE in the output contains local and global views of its light curve; auxiliary
features such as period and duration; and a label indicating whether the TCE is
consistent with being a transiting planet. The data sets produced by this script
can be used to train and evaluate models that classify Kepler TCEs.

The input TCEs and their associated labels are specified by the DR24 TCE Table,
which can be downloaded in CSV format from the NASA Exoplanet Archive at:

  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce

The downloaded CSV file should contain at least the following column names:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  tce_period: Orbital period of the detected event, in days.
  tce_time0bk: The time corresponding to the center of the first detected
      transit in Barycentric Julian Day (BJD) minus a constant offset of
      2,454,833.0 days.
  tce_duration: Duration of the detected transit, in hours.
  av_training_set: Autovetter training set label; one of PC (planet candidate),
      AFP (astrophysical false positive), NTP (non-transiting phenomenon),
      UNK (unknown).

The Kepler light curves can be downloaded from the Mikulski Archive for Space
Telescopes (MAST) at:

  http://archive.stsci.edu/pub/kepler/lightcurves.

The Kepler data is assumed to reside in a directory with the same structure as
the MAST archive. Specifically, the file names for a particular Kepler target
star should have the following format:

    .../${kep_id:0:4}/${kep_id}/kplr${kep_id}-${quarter_prefix}_${type}.fits,

where:
  kep_id is the Kepler id left-padded with zeros to length 9;
  quarter_prefix is the file name quarter prefix;
  type is one of "llc" (long cadence light curve) or "slc" (short cadence light
    curve).

The output TFRecord file contains one serialized tensorflow.train.Example
protocol buffer for each TCE in the input CSV file. Each Example contains the
following light curve representations:
  global_view: Vector of length 2001; the Global View of the TCE.
  local_view: Vector of length 201; the Local View of the TCE.

In addition, each Example contains the value of each column in the input TCE CSV
file. Some of these features may be useful as auxiliary features to the model.
The columns include:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  av_training_set: Autovetter training set label.
  tce_period: Orbital period of the detected event, in days.
  ...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.data import preprocess

input_tce_csv_file = "FinalProject/Data/tce_dr24.csv"
kepler_data_dir="FinalProject/Data/kepler/"
output_dir = "FinalProject/Data/tfrecord"

num_train_shards=8
num_worker_processes=5

# Name and values of the column in the input CSV file to use as training labels.
_LABEL_COLUMN = "av_training_set"
_ALLOWED_LABELS = {"PC", "AFP", "NTP"}


def _process_tce(tce):
  """Processes the light curve for a Kepler TCE and returns an Example proto.

  Args:
    tce: Row of the input TCE table.

  Returns:
    A tensorflow.train.Example proto containing TCE features.
  """
  all_time, all_flux = preprocess.read_light_curve(tce.kepid,
                                                   kepler_data_dir)
  time, flux = preprocess.process_light_curve(all_time, all_flux)
  return preprocess.generate_example_for_tce(time, flux, tce)


def _process_file_shard(tce_table, file_name):
  """Processes a single file shard.

  Args:
    tce_table: A Pandas DateFrame containing the TCEs in the shard.
    file_name: The output TFRecord file.
  """
  process_name = multiprocessing.current_process().name
  shard_name = os.path.basename(file_name)
  shard_size = len(tce_table)
  print(f"{process_name}: Processing {shard_size} items in shard {shard_name}")

  with tf.io.TFRecordWriter(file_name) as writer:
    num_processed = 0
    for _, tce in tce_table.iterrows():
      example = _process_tce(tce)
      if example is not None:
        writer.write(example.SerializeToString())

      num_processed += 1
      if not num_processed % 10:
        print(f"{process_name}: Processed {num_processed}/{shard_size} items in shard {shard_name}")

  print(f"{process_name}: Wrote {shard_size} items in shard {shard_name}")


def main():

  # Read CSV file of Kepler KOIs.
  tce_table = pd.read_csv(
      input_tce_csv_file, index_col="loc_rowid", comment="#")
  tce_table["tce_duration"] /= 24  # Convert hours to days.
  print(f"Read TCE CSV file with {len(tce_table)} rows.")

  # Filter TCE table to allowed labels.
  allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
  tce_table = tce_table[allowed_tces]
  num_tces = len(tce_table)
  print(f"Filtered to {num_tces} TCEs with labels in {list(_ALLOWED_LABELS)}.")

  # Randomly shuffle the TCE table.
  np.random.seed(123)
  tce_table = tce_table.iloc[np.random.permutation(num_tces)]
  print("Randomly shuffled TCEs.")

  # Partition the TCE table as follows:
  #   train_tces = 80% of TCEs
  #   val_tces = 10% of TCEs (for validation during training)
  #   test_tces = 10% of TCEs (for final evaluation)
  train_cutoff = int(0.80 * num_tces)
  val_cutoff = int(0.90 * num_tces)
  train_tces = tce_table[0:train_cutoff]
  val_tces = tce_table[train_cutoff:val_cutoff]
  test_tces = tce_table[val_cutoff:]
  print(f"Partitioned {num_tces} TCEs into training ({len(train_tces)}), validation ({len(val_tces)}) and test ({len(test_tces)})")

  # Further split training TCEs into file shards.
  file_shards = []  # List of (tce_table_shard, file_name).
  boundaries = np.linspace(0, len(train_tces),
                           num_train_shards + 1).astype(np.int)
  for i in range(num_train_shards):
    start = boundaries[i]
    end = boundaries[i + 1]
    filename = os.path.join(
        output_dir, "train-{:05d}-of-{:05d}".format(
            i, num_train_shards))
    file_shards.append((train_tces[start:end], filename))

  # Validation and test sets each have a single shard.
  file_shards.append((val_tces,
                      os.path.join(output_dir, "val-00000-of-00001")))
  file_shards.append((test_tces,
                      os.path.join(output_dir, "test-00000-of-00001")))
  num_file_shards = len(file_shards)

  # Launch subprocesses for the file shards.
  num_processes = min(num_file_shards, num_worker_processes)
  print(f"Launching {num_processes} subprocesses for {num_file_shards} total file shards")

  pool = multiprocessing.Pool(processes=num_processes)
  async_results = [
      pool.apply_async(_process_file_shard, file_shard)
      for file_shard in file_shards
  ]
  pool.close()

  # Instead of pool.join(), we call async_result.get() to ensure any exceptions
  # raised by the worker processes are also raised here.
  for async_result in async_results:
    async_result.get()

  print(f"Finished processing {num_file_shards} total file shards")


if __name__ == "__main__":
  main()
