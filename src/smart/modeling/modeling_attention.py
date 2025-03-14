import torch
from torch import nn


#######################################################
# Bag Attention Models
# head = 'att_att': BagAttention_Attention
# head = 'att': BagAttention
# head = 'one': BagOne
# head = 'origin': BagOriginAttention
# head = 'avg': BagAverage
#######################################################

class BagAttention_Attention(nn.Module):
    def __init__(self, pooling_dim, classifier, sfmx_t):
        super(BagAttention_Attention, self).__init__()
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
        self.sfmx_t = sfmx_t

    def forward(self, features, picked_label, similarity, bag_attention_target, write_attention=False):
        """
        name pooling_dim with `H`

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        # (N, H)
        # features = torch.stack(features)
        # (H, R)
        # att_mat = self.fc.weight.transpose(0, 1)

        # (H, R) * (H, 1) -> (H, R)
        # att_mat = att_mat * self.diag.unsqueeze(1)

        # (N, H) x (H, R) -> (N, R)
        # att_score = torch.matmul(features, att_mat)
        # features = features
        att_score = self.fc(features) * similarity

        # (N, R) -> (R, N)
        # print(f'temp is {sfmx_t}')
        softmax_att_score = self.softmax(att_score.transpose(0, 1) / self.sfmx_t)

        # (R, N) x (N, H) -> (R, H)
        feature_for_each_rel = torch.matmul(softmax_att_score, features)

        # (R, H) -> (R, R) -> (R)
        bag_logits = self.fc(feature_for_each_rel).diagonal().contiguous()

        if not self.training:
            bag_logits = self.softmax(bag_logits)
        attention_ = self.discriminator(features).squeeze(-1)
        # print(attention_, attention_.size())
        attention_loss = torch.nn.BCEWithLogitsLoss()(attention_,
                                                      bag_attention_target.to(att_score.device).float())

        return bag_logits, attention_loss


#######################################################

class BagAttention(nn.Module):
    def __init__(self, pooling_dim, classifier, sfmx_t):
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

        self.sfmx_t = sfmx_t

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
            softmax_att_score = self.softmax(att_score.transpose(0, 1) / self.sfmx_t)

            # (R, N) x (N, H) -> (R, H)
            feature_for_each_rel = torch.matmul(softmax_att_score, features)

            # (R, H) -> (R, R) -> (R)
            bag_logits = self.fc(feature_for_each_rel).diagonal().contiguous()

            if not self.training:
                bag_logits = self.softmax(bag_logits)

        attention_loss = torch.nn.BCEWithLogitsLoss()(self.discriminator(features).squeeze(-1),
                                                      bag_attention_target.to(att_score.device).float())

        return bag_logits, attention_loss


#######################################################

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


#######################################################

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


#######################################################

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


#######################################################

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values


class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, classifier):
        super(SelfAttentionClassifier, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.classifier = classifier

    def forward(self, x):
        attended_values = self.attention(x)
        x = attended_values.mean(dim=1)
        x = self.classifier(x)
        bag_logits = torch.sum(x, dim=0)
        if not self.training:
            bag_logits = self.softmax(bag_logits)
        return bag_logits


##################################################
# Gate Fusion Network
class GatedFusionNetwork(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_cls * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, logits1, logits2):
        # logits1/2 形状: (batch_size, num_cls)
        concat_logits = torch.cat([logits1, logits2], dim=-1)
        gate = self.gate_net(concat_logits)  # (batch_size, 1)
        fused_logits = gate * logits1 + (1 - gate) * logits2
        return fused_logits

#######################################################
# Bag Similarity Models
# simi = 'clip': ClipSimilarity
# simi = 'att': BagOriginAttention
# Cross Instances Similarity
class ClipSimilarity(nn.Module):
    def __init__(self, pooling_dim):
        super(ClipSimilarity, self).__init__()
        self.pooling_dim = pooling_dim
        self.projection = nn.Linear(self.pooling_dim, 2054 * 2)

    def forward(self, pair_img_feat, caption_feats):
        """

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        pair_img_feat = self.projection(pair_img_feat)
        sub_img_feat = pair_img_feat[:, :2054]
        obj_img_feat = pair_img_feat[:, 2054:]
        sub_caption_feat = caption_feats[:, 0, :]
        obj_caption_feat = caption_feats[:, 1, :]
        sub_img_feat /= sub_img_feat.norm(dim=-1, keepdim=True)
        obj_img_feat /= obj_img_feat.norm(dim=-1, keepdim=True)
        sub_caption_feat /= sub_caption_feat.norm(dim=-1, keepdim=True)
        obj_caption_feat /= obj_caption_feat.norm(dim=-1, keepdim=True)
        # The sub_caption confidence in a bag
        similarity1 = (sub_caption_feat @ sub_img_feat.T).sum(dim=0)
        # The obj_caption confidence in a bag
        similarity2 = (obj_caption_feat @ obj_img_feat.T).sum(dim=0)
        similarity = (similarity1 + similarity2).softmax(dim=-1).unsqueeze(1)

        return similarity

#######################################################
# Customize Attention Module here
class MyAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls, sfmx_t):
        super(MyAttentionClassifier, self).__init__()
        # self.MHA = MultiHeadSelfAttention(embed_dim, num_heads)
        # self.MLA = MultiHeadLatentAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.SelfAttention = SelfAttention(embed_dim[0][0])
        self.Similarity = ClipSimilarity(embed_dim[0][1] // 2)
        self.classifier1 = nn.Sequential(
            nn.Linear(embed_dim[0][1], embed_dim[0][1] * 2),
            nn.ReLU(),
            nn.Linear(embed_dim[0][1] * 2, num_cls)
        )
        self.BagAttention = BagAttention_Attention(embed_dim[0][1], self.classifier1, sfmx_t)
        self.classifier2 = nn.Sequential(
            nn.Linear(embed_dim[0][0], embed_dim[0][0]),
            nn.ReLU(),
            nn.Linear(embed_dim[0][0], num_cls)
        )
        self.Fusion = GatedFusionNetwork(num_cls)
        self.temperature = nn.Parameter(torch.ones(1, device=next(self.parameters()).device))

    def forward(self, caption_feats, pair_feats, label, pair_attention_label):
        sub_img_feat = pair_feats[:, :768]
        obj_img_feat = pair_feats[:, 768 * 2: 768 * 3]
        pair_img_feat = torch.cat([sub_img_feat, obj_img_feat], dim=-1)

        similarity = self.Similarity(pair_img_feat, caption_feats)
        bag_logits1, attention_loss = self.BagAttention(pair_feats, label, similarity, pair_attention_label)
        attended_values = self.SelfAttention(caption_feats[:, 0:2, :])
        x = attended_values.mean(dim=1)  # torch.Size([35, 2054])
        x = x * similarity
        x = self.classifier2(x)
        bag_logits2 = torch.sum(x, dim=0)
        bag_logits1 = bag_logits1 / self.temperature
        bag_logits2 = bag_logits2 / self.temperature
        bag_logits = self.Fusion(bag_logits1, bag_logits2)
        if not self.training:
            bag_logits = nn.functional.softmax(bag_logits, dim=-1)
        return bag_logits, attention_loss