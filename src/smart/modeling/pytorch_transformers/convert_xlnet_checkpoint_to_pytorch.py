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
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch

from smart.modeling.pytorch_transformers.modeling_xlnet import (CONFIG_NAME, WEIGHTS_NAME,
                                                    XLNetConfig,
                                                    XLNetLMHeadModel, XLNetForQuestionAnswering,
                                                    XLNetForSequenceClassification,
                                                    load_tf_weights_in_xlnet)

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

import logging
logging.basicConfig(level=logging.INFO)

def convert_xlnet_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path, finetuning_task=None):
    # Initialise PyTorch model
    config = XLNetConfig.from_json_file(bert_config_file)

    finetuning_task = finetuning_task.lower() if finetuning_task is not None else ""
    if finetuning_task in GLUE_TASKS_NUM_LABELS:
        print("Building PyTorch XLNetForSequenceClassification model from configuration: {}".format(str(config)))
        config.finetuning_task = finetuning_task
        config.num_labels = GLUE_TASKS_NUM_LABELS[finetuning_task]
        model = XLNetForSequenceClassification(config)
    elif 'squad' in finetuning_task:
        config.finetuning_task = finetuning_task
        model = XLNetForQuestionAnswering(config)
    else:
        model = XLNetLMHeadModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_xlnet(model, config, tf_checkpoint_path)

    # Save pytorch-model
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
    print("Save PyTorch model to {}".format(os.path.abspath(pytorch_weights_dump_path)))
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print("Save configuration file to {}".format(os.path.abspath(pytorch_config_dump_path)))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the TensorFlow checkpoint path.")
    parser.add_argument("--xlnet_config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained XLNet model. \n"
                               "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the folder to store the PyTorch model or dataset/vocab.")
    parser.add_argument("--finetuning_task",
                        default = None,
                        type = str,
                        help = "Name of a task on which the XLNet TensorFloaw model was fine-tuned")
    args = parser.parse_args()
    print(args)

    convert_xlnet_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                        args.xlnet_config_file,
                                        args.pytorch_dump_folder_path,
                                        args.finetuning_task)
