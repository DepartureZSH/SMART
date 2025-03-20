# Copyright (c) 2022 THUNLP Lab Tsinghua University. Licensed under the MIT license.
# This file contains implementation of some existing multi-instance-learning baselines and CLEVER
# Author: Tianyu Yu
# Data: 2022-09

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math

import torch
from torch import nn
from smart.utils.comm import get_rank
from smart.modeling.modeling_bert import BertImgModel
from smart.modeling.pytorch_transformers.modeling_bert import BertPreTrainedModel
from smart.modeling.modeling_attention import *
import json
from smart.modeling import NC, RED, GREEN, LIGHT_BLUE, LIGHT_PURPLE
from smart.modeling import sfmx_t, attention_w, head, simi, select_size, loss_w_t, loss_weight_mapping
###############################################################################
logger = logging.getLogger(__name__)

#######################################################

def BagLoss(rel_logits, rel_label, loss_weight):
    # criterion = nn.CrossEntropyLoss(loss_weight.to(rel_logits.device))
    criterion = nn.CrossEntropyLoss()
    # print(f'In loss {rel_logits.unsqueeze(0).shape} {rel_label.shape}')
    loss = criterion(rel_logits.unsqueeze(0), rel_label)
    if loss_w_t != -1.0:
        loss = loss / (loss_weight_mapping[rel_label[0].item()]) ** loss_w_t
    return loss

def forward_bag_pair_as_unit(model, label, input_ids, caption_feats, caption_feats_v2, image_caption_feats_v2, token_type_ids, attention_mask, img_feats,
                             object_box_lists, object_name_positions_lists, training, attention_label,
                             get_feat=False):
    # get output feat for images in this bag
    shard_size = 50 if training else 50
    num_shard = math.ceil(input_ids.shape[0] / shard_size)
    outputs = []

    for i in range(num_shard):
        outputs.append(model.bert(input_ids[shard_size * i: shard_size * (i + 1)],
                                  token_type_ids=token_type_ids[shard_size * i: shard_size * (i + 1)],
                                  attention_mask=attention_mask[shard_size * i: shard_size * (i + 1)],
                                  img_feats=img_feats[shard_size * i: shard_size * (i + 1)])[0])
    sequence_output = torch.cat(outputs, dim=0)

    num_object_of_images = [len(lst) for lst in object_box_lists]
    img_hidden = model._get_image_hidden(sequence_output, num_object_of_images)

    assert len(object_name_positions_lists) == len(num_object_of_images)
    object_name_hidden = model._get_obj_name_hidden(sequence_output, object_name_positions_lists)

    pair_img_feat = []
    pair_feat = []
    pair_attention_label = []
    for image_idx in range(len(object_box_lists)):
        sub_img_feat = img_hidden[image_idx][0]
        sub_name_feat = object_name_hidden[image_idx][0]
        sub_feat = torch.cat([sub_img_feat, sub_name_feat], dim=-1)

        obj_img_feat = img_hidden[image_idx][1]
        obj_name_feat = object_name_hidden[image_idx][1]
        obj_feat = torch.cat([obj_img_feat, obj_name_feat], dim=-1)
        pair_img_feat.append(torch.cat([sub_img_feat, obj_img_feat], dim=-1))
        pair_feat.append(torch.cat([sub_feat, obj_feat], dim=-1))
        pair_attention_label.append(attention_label[image_idx])

    pair_feat = torch.stack(pair_feat)
    pair_img_feat = torch.stack(pair_img_feat)
    pair_attention_label = torch.tensor(pair_attention_label)
    if get_feat:
        return pair_feat

    bag_logits, attention_loss = model.Attention(caption_feats_v2, image_caption_feats_v2, pair_feat, label, pair_attention_label)
    
    return bag_logits, attention_loss

class BagModel(BertPreTrainedModel):
    def __init__(self, config, head='Custom'):
        super(BagModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)

        # self.dropout = nn.Dropout(0.1)

        feat_size = config.hidden_size * 4
        caption_feats_size = config.img_feature_dim
        num_cls = 101

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, feat_size * 2),
            nn.ReLU(),
            nn.Linear(feat_size * 2, num_cls)
        )

        if head == 'Custom':
            self.Attention = MyAttentionClassifier([[caption_feats_size, feat_size]], 13, num_cls, sfmx_t)
        else:
            raise NotImplementedError

    def forward(self, bag_input_ids, bag_caption_feats, bag_token_type_ids=None, bag_attention_mask=None,
                position_ids=None, head_mask=None, bag_img_feats=None, bag_object_box_lists=None,
                bag_object_name_positions_lists=None, bag_head_obj_idxs_list=None, bag_tail_obj_idxs_list=None,
                bag_labels=None, attention_label_list=None, bag_image_ids_list=None, bag_key_list=None,
                preload_ids_list=None):
        logits_list = []
        loss_list = []

        bag_image_files, bag_caption_feats_v2, bag_image_cap_feats_v2 = preload_ids_list
        # bag_caption_feats_v2 
        # bag_image_cap_feats_v2 # torch.size([50, 1, 768])
        # process one bag at each iteration
        for bag_idx in range(len(bag_labels)):
            label = bag_labels[bag_idx]
            input_ids = bag_input_ids[bag_idx]
            caption_feats = bag_caption_feats[bag_idx] # torch.size([bs, 50, 2054])
            caption_feats_v2 = bag_caption_feats_v2[bag_idx] # torch.size([bs, 50, 768])
            caption_img_feats_v2 = bag_image_cap_feats_v2[bag_idx] # torch.size([bs, 1, 768])
            token_type_ids = bag_token_type_ids[bag_idx]
            attention_mask = bag_attention_mask[bag_idx]
            img_feats = bag_img_feats[bag_idx]
            object_box_lists = bag_object_box_lists[bag_idx]
            object_name_positions_lists = bag_object_name_positions_lists[bag_idx]
            attention_label = attention_label_list[bag_idx]

            if self.training:
                bag_logits, attention_loss = forward_bag_pair_as_unit(
                    model=self, label=label, input_ids=input_ids, caption_feats=caption_feats, 
                    caption_feats_v2=caption_feats_v2,
                    image_caption_feats_v2=caption_img_feats_v2,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask, img_feats=img_feats, object_box_lists=object_box_lists,
                    object_name_positions_lists=object_name_positions_lists, attention_label=attention_label,
                    training=True)
                bag_loss = BagLoss(bag_logits, label.unsqueeze(0), None)

                w = attention_w if head == 'att' else 0
                sum_loss = w * attention_loss + (1 - w) * bag_loss
                loss_list.append(sum_loss)
            else:
                bag_logits, attention_loss = forward_bag_pair_as_unit(
                    model=self, label=label, input_ids=input_ids, caption_feats=caption_feats,
                    caption_feats_v2=caption_feats_v2,
                    image_caption_feats_v2=caption_img_feats_v2,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask, img_feats=img_feats, object_box_lists=object_box_lists,
                    object_name_positions_lists=object_name_positions_lists, attention_label=attention_label,
                    training=False)
                # return torch.stack([pair_scores.sum(0) / len(pair_scores) for pair_scores in all_pair_scores])
                logits_list.append(bag_logits)

        return sum(loss_list) / len(loss_list) if self.training else torch.stack(logits_list)

    def _get_image_hidden(self, sequence_output, num_obj_of_images):
        outputs = []
        for seq, num_obj in zip(sequence_output, num_obj_of_images):
            outputs.append(seq[70:70 + num_obj])
        return outputs

    def _get_obj_name_hidden(self, sequence_output, object_name_positions_lists):
        outputs = []
        for seq, object_name_positions in zip(sequence_output, object_name_positions_lists):
            name_feats = []
            for object_name_pos in object_name_positions:
                object_name_feat = seq[object_name_pos].sum(dim=0) / len(object_name_pos)
                name_feats.append(object_name_feat)
            outputs.append(torch.stack(name_feats))
        return outputs
