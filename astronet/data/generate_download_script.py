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

r"""Generates a bash script for downloading light curves.

The input to this script is a CSV file of Kepler targets, for example the DR24
TCE table, which can be downloaded in CSV format from the NASA Exoplanet Archive
at:

  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce

Example usage:
  python generate_download_script.py \
    --kepler_csv_file=dr24_tce.csv \
    --download_dir=${HOME}/astronet/kepler
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import stat



_WGET_CMD = ("wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -erobots=off "
             "-R 'index*' -A _llc.fits")
_BASE_URL = "http://archive.stsci.edu/pub/kepler/lightcurves"


def main():
  kepler_csv_file = "../../FinalProject/Data/tce_dr24.csv"
  output_file = "../../FinalProject/Data/download_kepler.sh"
  download_dir = "../../FinalProject/Data/kepler"
  # Read Kepler targets.
  kepids = set()
  with open(kepler_csv_file) as f:
    reader = csv.DictReader(row for row in f if not row.startswith("#"))
    for row in reader:
      kepids.add(row["kepid"])

  num_kepids = len(kepids)

  # Write wget commands to script file.
  with open(output_file, "w") as f:
    f.write("#!/bin/sh\n")
    f.write("echo 'Downloading {} Kepler targets to {}'\n".format(
        num_kepids, download_dir))
    for i, kepid in enumerate(kepids):
      download_dir = "kepler"
      if i and not i % 10:
        f.write("echo 'Downloaded {}/{}'\n".format(i, num_kepids))
      kepid = "{0:09d}".format(int(kepid))  # Pad with zeros.
      subdir = "{}/{}".format(kepid[0:4], kepid)
      download_dir = os.path.join(download_dir, subdir)
      url = "{}/{}/".format(_BASE_URL, subdir)
      f.write("{} -P {} {}\n".format(_WGET_CMD, download_dir, url))

    f.write("echo 'Finished downloading {} Kepler targets to {}'\n".format(
        num_kepids, download_dir))

  # Make the download script executable.
  os.chmod(output_file, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

  print("{} Kepler targets will be downloaded to {}".format(
      num_kepids, output_file))
  print("To start download, run:\n  {}".format("./" + output_file
                                               if "/" not in output_file
                                               else output_file))


if __name__ == "__main__":
  main()
