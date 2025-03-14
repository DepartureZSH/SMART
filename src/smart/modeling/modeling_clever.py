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

logger = logging.getLogger(__name__)

from smart.modeling import NC, RED, GREEN, LIGHT_BLUE, LIGHT_PURPLE
from smart.modeling import sfmx_t, attention_w, head, simi, select_size, loss_w_t, loss_weight_mapping

class VRDBaselineModel(BertPreTrainedModel):
    def __init__(self, config):
        super(VRDBaselineModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)

        self.dropout = nn.Dropout(0.1)

        feat_size = config.hidden_size * 4
        num_cls = 101

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, feat_size * 2),
            nn.ReLU(),
            nn.Linear(feat_size * 2, num_cls)
        )

        self.classifier.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, relation_list=None,
                position_ids=None, head_mask=None, img_feats=None, object_box_lists=None,
                object_name_positions_lists=None, rel_labels_list=None, pairs_list=None):
        """

        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param relation_list: List of [[s, o, p], ...], each [[s, o, p], ...] contains relations labels
                              of one image
        :param position_ids:
        :param head_mask:
        :param img_feats:
        :param object_box_lists:
        :param object_name_positions_lists:
        :param rel_labels_list:
        :param pairs_list:
        :return:
        """
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        # [ [CLS]-emb, tok-0-emb, tok-1-emb, ..., obj-0-emb, obj-1-emb, ...]
        sequence_output = outputs[0]

        num_object_of_images = [len(lst) for lst in object_box_lists]
        img_hidden = self._get_image_hidden(sequence_output, num_object_of_images)

        assert len(object_name_positions_lists) == len(num_object_of_images)
        object_name_hidden = self._get_obj_name_hidden(sequence_output, object_name_positions_lists)

        labels = []
        logits = []

        for object_img_features, object_name_features, relations, rel_labels, pair_idxs in zip(
                img_hidden, object_name_hidden, relation_list, rel_labels_list, pairs_list):
            relation_pairs = {(s, o): p for s, o, p in relations}

            object_features = torch.cat([object_img_features, object_name_features], dim=-1)

            if self.training:
                # print(f'Training {pair_idxs.shape}')
                # [N, 768 * 4], where N is num of pairs
                img_pair_features = object_features[pair_idxs].view(pair_idxs.size(0), -1)

                # [N, 101]
                img_rel_logits = self.classifier(img_pair_features)

                logits.append(img_rel_logits)
                labels.append(rel_labels)
            else:
                for s in range(len(object_img_features)):
                    for o in range(len(object_img_features)):
                        if o == s:
                            continue
                        if (s, o) in relation_pairs:
                            p = relation_pairs[(s, o)]
                        else:
                            p = 0  # no-relation
                        s_feat = object_features[s]
                        o_feat = object_features[o]
                        pair_feat = torch.cat([s_feat, o_feat], dim=-1)
                        logits.append(self.classifier(pair_feat))
                        labels.append(p)
        if self.training:
            logits = torch.cat(logits, dim=0)
            labels = torch.cat(labels, dim=0)
        else:
            logits = torch.stack(logits)
            labels = torch.tensor(labels).to(logits.device)

        loss = None
        if self.training:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

        scores = logits.softmax(-1)
        return loss, scores, labels

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


class BagAttention(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagAttention, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101

        self.fc = classifier
        self.softmax = nn.Softmax(-1)
        self.diag = nn.Parameter(torch.ones(self.pooling_dim))
        self.drop = nn.Dropout(p=0.2)

        self.discriminator = nn.Sequential(
            nn.Linear(self.pooling_dim, self.pooling_dim),
            nn.ReLU(),
            nn.Linear(self.pooling_dim, 1)
        )

    def forward(self, features, picked_label, bag_attention_target, write_attention=False):
        """
        name pooling_dim with `H`

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        # (N, H)
        # features = torch.stack(features)

        if self.training and False:
            # (1, H)
            # att_mat = self.fc.weight[picked_label].unsqueeze(0)
            # assert att_mat.shape == (1, self.pooling_dim + 400), f'{att_mat.shape}, {(1, self.pooling_dim)}'
            #
            # # (1, H) * (1, H) -> (1, H)
            # att_mat = att_mat * self.diag.unsqueeze(0)
            # assert att_mat.shape == (1, self.pooling_dim + 400), f'{att_mat.shape}, {(1, self.pooling_dim)}'

            # (N, H) * (1, H) -> (N, H) -> (N)
            # att_score = (features * att_mat).sum(-1)
            att_score = self.fc(features)[:, picked_label]
            assert att_score.shape == (len(features),), f'{att_score.shape}, {(len(features),)}'

            # (N) -> (N)
            softmax_att_score = self.softmax(att_score / sfmx_t)
            # if random.random() < 0.01:
            #     msg = f'{RED}Train: {softmax_att_score} {GREEN}{bag_attention_target}{NC}'
            #     print(msg)
            #     logging.info(msg)

            # (N, 1) * (N, H) -> (N, H) -> (H)
            bag_feature = (softmax_att_score.unsqueeze(-1) * features).sum(0)
            # assert torch.equal(bag_feature, features[0]) # Test bag-size 1

            # (H)
            bag_feature = self.drop(bag_feature)

            # (R)
            bag_logits = self.fc(bag_feature)
            # if random.random() < 0.001:
            #     v = bag_logits.softmax(-1)
            #     msg = f'{RED}Train logits: {NC}[' + ', '.join(
            #         map(lambda x: f'{int(x * 1000)}',
            #             v.tolist())) + f'] {GREEN}{int(v[picked_label].item() * 1000)}{NC}'
            #     print(msg)
        else:
            # (H, R)
            # att_mat = self.fc.weight.transpose(0, 1)

            # (H, R) * (H, 1) -> (H, R)
            # att_mat = att_mat * self.diag.unsqueeze(1)

            # (N, H) x (H, R) -> (N, R)
            # att_score = torch.matmul(features, att_mat)
            att_score = self.fc(features)

            # (N, R) -> (R, N)
            # print(f'temp is {sfmx_t}')
            softmax_att_score = self.softmax(att_score.transpose(0, 1) / sfmx_t)

            # (R, N) x (N, H) -> (R, H)
            feature_for_each_rel = torch.matmul(softmax_att_score, features)

            # (R, H) -> (R, R) -> (R)
            bag_logits = self.fc(feature_for_each_rel).diagonal().contiguous()

            if not self.training:
                bag_logits = self.softmax(bag_logits)

        attention_loss = torch.nn.BCEWithLogitsLoss()(self.discriminator(features).squeeze(-1),
                                                      bag_attention_target.to(att_score.device).float())

        return bag_logits, attention_loss


class BagOne(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagOne, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101

        self.fc = classifier
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(p=0.2)

        self.discriminator = nn.Sequential(
            nn.Linear(self.pooling_dim, self.pooling_dim),
            nn.ReLU(),
            nn.Linear(self.pooling_dim, 1)
        )

    def forward(self, features, picked_label, bag_attention_target):
        """

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """

        if self.training:
            # (N, R)
            instance_scores = self.fc(features).softmax(dim=-1)

            # (N, R) -> (N, ) -> 1
            max_index = instance_scores[:, picked_label].argmax()

            # (N, H) -> (H, )
            bag_rep = features[max_index]

            # (H, ) -> (R, )
            bag_logits = self.fc(self.drop(bag_rep))
        else:
            # (N, R)
            instance_scores = self.fc(features).softmax(dim=-1)

            # (N, R) -> (R, )
            score_for_each_rel = instance_scores.max(dim=0)[0]

            bag_logits = score_for_each_rel

        assert bag_logits.shape == (self.num_rel_cls,)
        attention_loss = 0

        return bag_logits, attention_loss


class BagOriginAttention(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagOriginAttention, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101

        self.fc = classifier
        self.softmax = nn.Softmax(-1)
        self.diag = nn.Parameter(torch.ones(self.pooling_dim))
        self.drop = nn.Dropout(p=0.2)

        self.discriminator = nn.Sequential(
            nn.Linear(self.pooling_dim, self.pooling_dim),
            nn.ReLU(),
            nn.Linear(self.pooling_dim, 1)
        )

    def forward(self, features, picked_label, bag_attention_target):
        """
        name pooling_dim with `H`

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        # (N, H)
        # features = torch.stack(features)

        if self.training:
            att_score = self.fc(features)[:, picked_label]
            assert att_score.shape == (len(features),), f'{att_score.shape}, {(len(features),)}'

            # (N) -> (N)
            softmax_att_score = self.softmax(att_score)

            # (N, 1) * (N, H) -> (N, H) -> (H)
            bag_feature = (softmax_att_score.unsqueeze(-1) * features).sum(0)
            # assert torch.equal(bag_feature, features[0]) # Test bag-size 1

            # (H)
            bag_feature = self.drop(bag_feature)

            # (R)
            bag_logits = self.fc(bag_feature)
        else:
            att_score = self.fc(features)

            # (N, R) -> (R, N)
            softmax_att_score = self.softmax(att_score.transpose(0, 1))

            # (R, N) x (N, H) -> (R, H)
            feature_for_each_rel = torch.matmul(softmax_att_score, features)

            # (R, H) -> (R, R) -> (R)
            bag_logits = self.fc(feature_for_each_rel).diagonal().contiguous()

            if not self.training:
                bag_logits = self.softmax(bag_logits)

        attention_loss = torch.nn.BCEWithLogitsLoss()(self.discriminator(features).squeeze(-1),
                                                      bag_attention_target.to(att_score.device).float())

        return bag_logits, attention_loss


class BagAverage(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagAverage, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101
        self.drop = nn.Dropout(p=0.2)
        self.classifier = classifier

    def forward(self, features, picked_label, bag_attention_target):
        """

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        # features = self.encoder(features)

        # (N, H) -> (H, )
        mean = torch.mean(features, dim=0)
        mean = self.drop(mean)
        bag_logits = self.classifier(mean)
        if not self.training:
            return bag_logits.softmax(-1), 0
        return bag_logits, 0


def BagLoss(rel_logits, rel_label, loss_weight):
    # criterion = nn.CrossEntropyLoss(loss_weight.to(rel_logits.device))
    criterion = nn.CrossEntropyLoss()
    # print(f'In loss {rel_logits.unsqueeze(0).shape} {rel_label.shape}')
    loss = criterion(rel_logits.unsqueeze(0), rel_label)
    if loss_w_t != -1.0:
        loss = loss / (loss_weight_mapping[rel_label[0].item()]) ** loss_w_t
    return loss


def instanceMaxHead(pair_feats, classifier):
    shard_size = 1000
    num_shard = math.ceil(pair_feats.shape[0] / shard_size)
    scores = []
    for i in range(num_shard):
        logits = classifier(pair_feats[shard_size * i: shard_size * (i + 1)])  # (shard_size, 101)
        scores.append(logits.softmax(-1))
    scores = torch.cat(scores, dim=0).max(0)[0]
    return scores


def forward_bag_pair_as_unit(model, label, input_ids, token_type_ids, attention_mask, img_feats,
                             object_box_lists, object_name_positions_lists, training, attention_label,
                             get_feat=False):
    # get output feat for images in this bag
    shard_size = 50 if training else 50
    num_shard = math.ceil(input_ids.shape[0] / shard_size)
    outputs = []

    for i in range(num_shard):
        # [ [CLS]-emb, tok-0-emb, tok-1-emb, ..., obj-0-emb, obj-1-emb, ...]
        # (BagSize, SeqLength, 768)
        outputs.append(model.bert(input_ids[shard_size * i: shard_size * (i + 1)],
                                  token_type_ids=token_type_ids[shard_size * i: shard_size * (i + 1)],
                                  attention_mask=attention_mask[shard_size * i: shard_size * (i + 1)],
                                  img_feats=img_feats[shard_size * i: shard_size * (i + 1)])[0])
    sequence_output = torch.cat(outputs, dim=0)

    num_object_of_images = [len(lst) for lst in object_box_lists]
    img_hidden = model._get_image_hidden(sequence_output, num_object_of_images)

    assert len(object_name_positions_lists) == len(num_object_of_images)
    object_name_hidden = model._get_obj_name_hidden(sequence_output, object_name_positions_lists)

    pair_feat = []
    pair_attention_label = []
    for image_idx in range(len(object_box_lists)):
        sub_img_feat = img_hidden[image_idx][0]
        sub_name_feat = object_name_hidden[image_idx][0]
        sub_feat = torch.cat([sub_img_feat, sub_name_feat], dim=-1)

        obj_img_feat = img_hidden[image_idx][1]
        obj_name_feat = object_name_hidden[image_idx][1]
        obj_feat = torch.cat([obj_img_feat, obj_name_feat], dim=-1)
        pair_feat.append(torch.cat([sub_feat, obj_feat], dim=-1))
        pair_attention_label.append(attention_label[image_idx])

    pair_feat = torch.stack(pair_feat)
    pair_attention_label = torch.tensor(pair_attention_label)
    if get_feat:
        return pair_feat

    bag_logits, attention_loss = model.head(pair_feat, label, pair_attention_label)

    return bag_logits, attention_loss


sorted_test_image_ids = {}


def process_image_ids(image_ids_raw, pair_image_idx, sorted_pair_belonging_image_idxs):
    sorted_pair_belonging_image_ids = [image_ids_raw[pair_image_idx[x]] for x in sorted_pair_belonging_image_idxs]
    sorted_image_ids = []
    collected_ids = set()
    for i in sorted_pair_belonging_image_ids:
        if i in collected_ids:
            continue
        collected_ids.add(i)
        sorted_image_ids.append(i)
    return sorted_image_ids


def select_images(model, bag_labels, bag_input_ids, bag_token_type_ids, bag_attention_mask, bag_img_feats,
                  bag_object_box_lists, bag_object_name_positions_lists, bag_head_obj_idxs_list, bag_tail_obj_idxs_list,
                  training, bag_image_ids_list, bag_key_list):
    all_selected_images = []
    all_pair_feat = []
    all_pair_img_idx = []
    all_pair_scores = []
    with torch.no_grad():
        # process one bag at each iteration
        for bag_idx in range(len(bag_labels)):
            pair_feat = []
            label = bag_labels[bag_idx]
            key = bag_key_list[bag_idx]
            input_ids = bag_input_ids[bag_idx]
            token_type_ids = bag_token_type_ids[bag_idx]
            attention_mask = bag_attention_mask[bag_idx]
            img_feats = bag_img_feats[bag_idx]
            object_box_lists = bag_object_box_lists[bag_idx]
            object_name_positions_lists = bag_object_name_positions_lists[bag_idx]
            head_obj_idxs_list = bag_head_obj_idxs_list[bag_idx]
            tail_obj_idxs_list = bag_tail_obj_idxs_list[bag_idx]
            image_ids_raw = bag_image_ids_list[bag_idx]

            # print(f'input_ids.shape={input_ids.shape}, img_feat.shape={img_feats.shape}')

            # get output feat for images in this bag
            shard_size = 400
            num_shard = math.ceil(input_ids.shape[0] / shard_size)
            outputs = []
            for i in range(num_shard):
                # [ [CLS]-emb, tok-0-emb, tok-1-emb, ..., obj-0-emb, obj-1-emb, ...]
                # (BagSize, SeqLength, 768)
                outputs.append(model.bert(input_ids[shard_size * i: shard_size * (i + 1)],
                                          token_type_ids=token_type_ids[shard_size * i: shard_size * (i + 1)],
                                          attention_mask=attention_mask[shard_size * i: shard_size * (i + 1)],
                                          img_feats=img_feats[shard_size * i: shard_size * (i + 1)])[0])
            sequence_output = torch.cat(outputs, dim=0)

            num_object_of_images = [len(lst) for lst in object_box_lists]
            img_hidden = model._get_image_hidden(sequence_output, num_object_of_images)

            assert len(object_name_positions_lists) == len(num_object_of_images)
            object_name_hidden = model._get_obj_name_hidden(sequence_output, object_name_positions_lists)

            pair_image_idx = []
            for image_idx in range(len(object_box_lists)):
                sub_img_feat = torch.stack([img_hidden[image_idx][x] for x in head_obj_idxs_list[image_idx]])
                sub_name_feat = torch.stack([object_name_hidden[image_idx][x] for x in head_obj_idxs_list[image_idx]])
                sub_feat = torch.cat([sub_img_feat, sub_name_feat], dim=-1)

                obj_img_feat = torch.stack([img_hidden[image_idx][x] for x in tail_obj_idxs_list[image_idx]])
                obj_name_feat = torch.stack([object_name_hidden[image_idx][x] for x in tail_obj_idxs_list[image_idx]])
                obj_feat = torch.cat([obj_img_feat, obj_name_feat], dim=-1)

                for s in range(len(sub_feat)):
                    for o in range(len(obj_feat)):
                        if head_obj_idxs_list[image_idx][0] == tail_obj_idxs_list[image_idx][0] and s == o:
                            continue
                        pair_feat.append(torch.cat([sub_feat[s], obj_feat[o]], dim=-1))
                        pair_image_idx.append(image_idx)

            pair_feat = torch.stack(pair_feat)  # N, H
            pair_scores = model.classifier(pair_feat).softmax(-1)  # N, 101
            all_pair_scores.append(pair_scores)

            if training:
                sorted_indices = pair_scores[:, label].sort(descending=True)[1].tolist()
                assert len(sorted_indices) == len(pair_feat)

                selected_image_idxs = set()
                for i in sorted_indices:
                    selected_image_idxs.add(pair_image_idx[i])
                    if len(selected_image_idxs) == select_size:
                        break
                all_selected_images.append(selected_image_idxs)
            else:
                all_label_selected_image_idxs = []
                sorted_test_image_ids[key] = {}
                for label in range(101):
                    sorted_indices = pair_scores[:, label].sort(descending=True)[1].tolist()
                    sorted_test_image_ids[key][label] = process_image_ids(image_ids_raw, pair_image_idx, sorted_indices)
                    assert len(sorted_indices) == len(pair_feat)

                    selected_image_idxs = set()
                    for i in sorted_indices:
                        selected_image_idxs.add(pair_image_idx[i])
                        if len(selected_image_idxs) == select_size:
                            break
                    all_label_selected_image_idxs.append(selected_image_idxs)
                all_selected_images.append(all_label_selected_image_idxs)
                all_pair_feat.append(pair_feat)
                all_pair_img_idx.append(pair_image_idx)

    return all_selected_images, all_pair_feat, all_pair_img_idx, all_pair_scores


class BagModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BagModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)

        self.dropout = nn.Dropout(0.1)

        feat_size = config.hidden_size * 4
        num_cls = 101

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, feat_size * 2),
            nn.ReLU(),
            nn.Linear(feat_size * 2, num_cls)
        )

        if head == 'att':
            self.head = BagAttention(feat_size, self.classifier)
        elif head == 'avg':
            self.head = BagAverage(feat_size, self.classifier)
            print('Use Average bag head')
        elif head == 'origin_att':
            self.head = BagOriginAttention(feat_size, self.classifier)
        elif head == 'one':
            self.head = BagOne(feat_size, self.classifier)
        elif head == 'max':
            pass
        else:
            raise NotImplementedError

    def forward(self, bag_input_ids, bag_token_type_ids=None, bag_attention_mask=None,
                position_ids=None, head_mask=None, bag_img_feats=None, bag_object_box_lists=None,
                bag_object_name_positions_lists=None, bag_head_obj_idxs_list=None, bag_tail_obj_idxs_list=None,
                bag_labels=None, attention_label_list=None, bag_image_ids_list=None, bag_key_list=None,
                preload_ids_list=None):
        logits_list = []
        loss_list = []

        # process one bag at each iteration
        for bag_idx in range(len(bag_labels)):
            label = bag_labels[bag_idx]
            input_ids = bag_input_ids[bag_idx]
            token_type_ids = bag_token_type_ids[bag_idx]
            attention_mask = bag_attention_mask[bag_idx]
            img_feats = bag_img_feats[bag_idx]
            object_box_lists = bag_object_box_lists[bag_idx]
            object_name_positions_lists = bag_object_name_positions_lists[bag_idx]
            attention_label = attention_label_list[bag_idx]

            if self.training:
                bag_logits, attention_loss = forward_bag_pair_as_unit(
                    model=self, label=label, input_ids=input_ids, token_type_ids=token_type_ids,
                    attention_mask=attention_mask, img_feats=img_feats, object_box_lists=object_box_lists,
                    object_name_positions_lists=object_name_positions_lists, attention_label=attention_label,
                    training=True)
                bag_loss = BagLoss(bag_logits, label.unsqueeze(0), None)

                w = attention_w if head == 'att' else 0
                sum_loss = w * attention_loss + (1 - w) * bag_loss
                loss_list.append(sum_loss)
            else:
                if head == 'max':
                    pair_feat = forward_bag_pair_as_unit(
                        model=self, label=label, input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask, img_feats=img_feats, object_box_lists=object_box_lists,
                        object_name_positions_lists=object_name_positions_lists, attention_label=attention_label,
                        training=False, get_feat=True)
                    bag_logits = instanceMaxHead(pair_feat, self.classifier)
                else:
                    bag_logits, attention_loss = forward_bag_pair_as_unit(
                        model=self, label=label, input_ids=input_ids, token_type_ids=token_type_ids,
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