# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert OpenAI GPT checkpoint."""

from __future__ import absolute_import, division, print_function

import argparse
import json
from io import open

import torch
import numpy

from smart.modeling.pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME
from smart.modeling.pytorch_transformers.tokenization_xlm import VOCAB_FILES_NAMES

import logging
logging.basicConfig(level=logging.INFO)

def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    # Load checkpoint
    chkpt = torch.load(xlm_checkpoint_path, map_location='cpu')

    model = chkpt['model']

    config = chkpt['params']
    config = dict((n, v) for n, v in config.items() if not isinstance(v, (torch.FloatTensor, numpy.ndarray)))

    vocab = chkpt['dico_word2id']
    vocab = dict((s + '</w>' if s.find('@@') == -1 and i > 13 else s.replace('@@', ''), i) for s, i in vocab.items())

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    pytorch_vocab_dump_path = pytorch_dump_folder_path + '/' +  VOCAB_FILES_NAMES['vocab_file']

    print("Save PyTorch model to {}".format(pytorch_weights_dump_path))
    torch.save(model, pytorch_weights_dump_path)

    print("Save configuration file to {}".format(pytorch_config_dump_path))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2) + "\n")

    print("Save vocab file to {}".format(pytorch_config_dump_path))
    with open(pytorch_vocab_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--xlm_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the official PyTorch dump.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)
