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
# MHA
# Caption <relation, caption>
# More aspects

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                                          embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x


class MultiHeadSelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, classifier):
        super(MultiHeadSelfAttentionClassifier, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.classifier = classifier

    def forward(self, x):
        x = self.attention(x)
        x = x.mean(dim=1)  # 对每个位置的向量求平均
        x = self.classifier(x)
        bag_logits = torch.sum(x, dim=0)
        if not self.training:
            bag_logits = self.softmax(bag_logits)
        return bag_logits

##################################################
# MLA
# Less cost than MHA
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, heads, seq_len, dim]
    K, V: [batch, heads, seq_len, dim]
    mask: [batch, 1, seq_len, seq_len]
    """
    dim = Q.size(-1)

    # 计算注意力分数
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)

    # 应用掩码
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    # Softmax归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # 计算输出
    return torch.matmul(attn_weights, V)


class MultiHeadLatentAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=1024, rope_theta=10000.0):
        # dim 需要4的倍数
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dh = embed_dim // num_heads
        self.q_proj_dim = embed_dim // 2
        self.kv_proj_dim = (2 * embed_dim) // 3

        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh - self.qk_nope_dim

        ## Q projections
        # Lora
        self.W_dq = torch.nn.Parameter(0.01 * torch.randn((embed_dim, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01 * torch.randn((self.q_proj_dim, self.embed_dim)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)

        ## KV projections
        # Lora
        self.W_dkv = torch.nn.Parameter(0.01 * torch.randn((embed_dim, self.kv_proj_dim + self.qk_rope_dim)))
        self.W_ukv = torch.nn.Parameter(0.01 * torch.randn((self.kv_proj_dim,
                                                            self.embed_dim + (self.num_heads * self.qk_nope_dim))))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

        # output projection
        self.W_o = torch.nn.Parameter(0.01 * torch.randn((embed_dim, embed_dim)))

        # RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # https://github.com/lucidrains/rotary-embedding-torch/tree/main
        # visualize emb later to make sure it looks ok
        # we do self.dh here instead of self.qk_rope_dim because its better
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, features, kv_cache=None, past_length=0):
        """
        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        x = features
        B, S, D = x.size()

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, -1, self.num_heads, self.dh).transpose(1, 2)
        # print(Q.size())
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # print(self.cos_cached.size())
        # print(self.sin_cached.size())
        # Q Decoupled RoPE
        if self.qk_rope_dim % 2 == 0:
            cos_q = self.cos_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
            sin_q = self.sin_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        else:
            cos_q = self.cos_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2 + 1].repeat(1, 1, 1, 2)
            sin_q = self.sin_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2 + 1].repeat(1, 1, 1, 2)
        # print(Q_for_rope.size())
        # print(cos_q.size())
        # print(sin_q.size())
        Q_for_rope = self.apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                  [self.kv_proj_dim, self.qk_rope_dim],
                                                  dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)

        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.num_heads, self.dh + self.qk_nope_dim).transpose(1, 2)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)

        # K Rope
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1, 2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        K_for_rope = self.apply_rope_x(K_for_rope, cos_k, sin_k)

        # apply position encoding to each head
        K_for_rope = K_for_rope.repeat(1, self.num_heads, 1, 1)

        # split into multiple heads
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V  # already reshaped before the split

        # make attention mask
        mask = torch.ones((S, S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]

        sq_mask = mask == 1

        # attention
        x = scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            mask=sq_mask
        )

        x = x.transpose(1, 2).reshape(B, S, D)

        # apply projection
        x = x @ self.W_o.T

        return x, compressed_kv

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, q, k, cos, sin):
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

    def apply_rope_x(self, x, cos, sin):
        return (x * cos) + (self.rotate_half(x) * sin)


class MLAClassifier(nn.Module):
    def __init__(self, pooling_dim=2054, num_heads=13, classifier=None):
        super().__init__()
        # 输入投影层（可选：若原始特征维度不等于 embed_dim）
        # self.input_proj = nn.Linear(2054, embed_dim)

        # MLA 模块
        self.mla = MultiHeadLatentAttention(embed_dim=pooling_dim, num_heads=num_heads)

        # 分类头
        self.classifier = classifier

    def forward(self, x):
        # x 形状: [batch_size, seq_len=1, feature_dim=2054]
        # x = self.input_proj(x)  # 投影到 embed_dim，形状 [batch_size, 1, embed_dim]
        attended_x, _ = self.mla(x, kv_cache=None, past_length=0)  # MLA 输出 [batch_size, 1, embed_dim]
        x = attended_x.mean(dim=1)  # 全局平均池化，形状 [batch_size, embed_dim]
        bag_logits = self.classifier(x)  # 分类输出 [batch_size, 101]
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
        sub_img_feat = sub_img_feat / (sub_img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        obj_img_feat = obj_img_feat / (obj_img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        sub_caption_feat = sub_caption_feat / (sub_caption_feat.norm(dim=-1, keepdim=True) + 1e-8)
        obj_img_feat = obj_img_feat / (obj_img_feat.norm(dim=-1, keepdim=True) + 1e-8)
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


#######################################################
# MoE

class Expert1(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls):
        super(Expert1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim[0][1], embed_dim[0][1] * 2),
            nn.ReLU(),
            nn.Linear(embed_dim[0][1] * 2, num_cls)
        )
        self.BagAttention = BagAttention_Attention(embed_dim[0][1], self.classifier)

    def forward(self, similarity, caption_feats, pair_feats, label, pair_attention_label):
        bag_logits, _ = self.BagAttention(pair_feats, label, similarity, pair_attention_label)
        return bag_logits

class Expert2(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls):
        super(Expert2, self).__init__()
        self.SelfAttention = SelfAttention(embed_dim[0][0])
        self.projection = nn.Linear(embed_dim[0][0], embed_dim[0][1] // 2)
        self.MLA = MultiHeadLatentAttention(embed_dim=embed_dim[0][1] + embed_dim[0][1] // 2, num_heads=16)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim[0][1] + embed_dim[0][1] // 2, embed_dim[0][1]),
            nn.ReLU(),
            nn.Linear(embed_dim[0][1], num_cls)
        )

    def forward(self, similarity, caption_feats, pair_feats, label, pair_attention_label):
        x = self.SelfAttention(caption_feats[:, 0:2, :])
        x = x.mean(dim=1)  # torch.Size([35, 2054])
        x = x * similarity  # torch.Size([35, 2054])
        x = self.projection(x)
        x = torch.cat([pair_feats, x], dim=-1)
        x = x.view(x.shape[0], 1, x.shape[1])
        x, _ = self.MLA(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        bag_logits = torch.sum(x, dim=0)
        return bag_logits

class MyAttentionClassifier2(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls):
        super(MyAttentionClassifier2, self).__init__()
        self.experts = nn.ModuleList([Expert1(embed_dim, num_heads, num_cls), Expert2(embed_dim, num_heads, num_cls)])
        self.Similarity = ClipSimilarity(embed_dim[0][1] // 2)
        embed_dim = embed_dim[0][0] * 2 + embed_dim[0][1]
        self.gatting = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, caption_feats, pair_feats, label, pair_attention_label):
        sub_img_feat = pair_feats[:, :768]
        sub_tag_feat = pair_feats[:, 768: 768 * 2]
        obj_img_feat = pair_feats[:, 768 * 2: 768 * 3]
        obj_tag_feat = pair_feats[:, 768 * 3:]
        pair_img_feat = torch.cat([sub_img_feat, obj_img_feat], dim=-1)
        similarity = self.Similarity(pair_img_feat, caption_feats)
        values, indices = similarity.squeeze().topk(1)
        caption_feat = caption_feats[indices, 0:2, :].view(1, -1)
        pair_feat = pair_feats[indices, :].view(1, -1)
        feat = torch.cat([pair_feat, caption_feat], dim=-1)
        gates = self.gatting(feat)
        expert_outs = [expert(similarity, caption_feats, pair_feats, label, pair_attention_label) for expert in self.experts]
        weighted_outs = [gate * expert_out for gate, expert_out in zip(gates.unbind(1), expert_outs)]
        bag_logits = torch.stack(weighted_outs, dim=-1).sum(dim=-1).squeeze()
        if not self.training:
            bag_logits = nn.functional.softmax(bag_logits, dim=-1)
        return bag_logits, 0

#######################################################
# BLIP2-MLA-MoE

class SimilarityV3(nn.Module):
    def __init__(self, pooling_dim):
        super(SimilarityV3, self).__init__()
        self.pooling_dim = pooling_dim
        self.projection = nn.Linear(self.pooling_dim, self.pooling_dim)

    def forward(self, pair_img_feat, caption_feats):
        """

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        caption_feats = caption_feats.view(-1, self.pooling_dim)
        caption_feats = self.projection(caption_feats)
        sub_img_feat = pair_img_feat[:, :768]
        obj_img_feat = pair_img_feat[:, 768:]
        # sub_caption_feat = caption_feats[:, 0, :]
        # obj_caption_feat = caption_feats[:, 1, :]
        sub_img_feat = sub_img_feat / (sub_img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        obj_img_feat = obj_img_feat / (obj_img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        caption_feat = caption_feats / (caption_feats.norm(dim=-1, keepdim=True) + 1e-8)
        # The sub_caption confidence in a bag
        similarity1 = (caption_feat @ sub_img_feat.T).sum(dim=0)
        # The obj_caption confidence in a bag
        similarity2 = (caption_feat @ obj_img_feat.T).sum(dim=0)
        similarity = (similarity1 + similarity2).softmax(dim=-1).unsqueeze(1)

        return similarity

class MyAttentionClassifier4(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls, sfmx_t):
        super(MyAttentionClassifier4, self).__init__()
        # self.experts = nn.ModuleList([Expert1(embed_dim, num_heads, num_cls), Expert2(embed_dim, num_heads, num_cls)])
        self.Similarity = SimilarityV3(embed_dim[0][0])
        self.MLA = MultiHeadLatentAttention(embed_dim=embed_dim[0][0] + embed_dim[0][1], num_heads=16)
        self.classifier1 = nn.Sequential(
            nn.Linear(embed_dim[0][1], embed_dim[0][1] * 2),
            nn.ReLU(),
            nn.Linear(embed_dim[0][1] * 2, num_cls)
        )
        self.LayerNorm = nn.LayerNorm(embed_dim[0][0])
        self.BagAttention = BagAttention_Attention(embed_dim[0][1], self.classifier1, 1)
        self.SelfAttention = SelfAttention(embed_dim[0][0])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim[0][0] + embed_dim[0][1], embed_dim[0][1]),
            nn.ReLU(),
            nn.Linear(embed_dim[0][1], num_cls)
        )
        self.Fusion = GatedFusionNetwork(num_cls)
        self.temperature = nn.Parameter(torch.ones(1, device=next(self.parameters()).device))

    def forward(self, caption_feats_v2, image_caption_feats_v2, pair_feats, label, pair_attention_label):
        # caption_feats_v2 # torch.size([bs, 50, 768])
        # image_caption_feats_v2 # torch.size([bs, 1, 768])
        # pair_feats # torch.size([bs, 768*4])
        sub_img_feat = pair_feats[:, :768]
        sub_tag_feat = pair_feats[:, 768: 768 * 2]
        obj_img_feat = pair_feats[:, 768 * 2: 768 * 3]
        obj_tag_feat = pair_feats[:, 768 * 3:]
        pair_img_feat = torch.cat([sub_img_feat, obj_img_feat], dim=-1)
        pair_tag_feat = torch.cat([sub_tag_feat, obj_tag_feat], dim=-1)
        caption_feats = torch.cat([image_caption_feats_v2, caption_feats_v2], dim=1)

        similarity = self.Similarity(pair_img_feat, image_caption_feats_v2)
        bag_logits1, attention_loss = self.BagAttention(pair_feats, label, similarity, pair_attention_label)
        x = self.LayerNorm(caption_feats[:, 0:3, :])
        x = self.SelfAttention(x)
        x = x.mean(dim=1)  # torch.Size([35, 2054])
        x = x * similarity
        x = torch.cat([pair_feats, x], dim=-1)
        x = x.view(x.shape[0], 1, x.shape[1])
        x, _ = self.MLA(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        bag_logits2 = torch.sum(x, dim=0)
        bag_logits1 = bag_logits1 / self.temperature
        bag_logits2 = bag_logits2 / self.temperature
        bag_logits = self.Fusion(bag_logits1, bag_logits2)
        if not self.training:
            bag_logits = nn.functional.softmax(bag_logits, dim=-1)
        return bag_logits1, 0
    
#######################################################
# SiMilarity Attention RecurrenT Block
class SMART(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls):
        super(SMART, self).__init__()
        self.SelfAttention = SelfAttention(embed_dim[0][0])
        self.projection = nn.Linear(embed_dim[0][0], embed_dim[0][1] // 2)
        self.MLA = MultiHeadLatentAttention(embed_dim=embed_dim[0][1] + embed_dim[0][1] // 2, num_heads=16)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim[0][1] + embed_dim[0][1] // 2, embed_dim[0][1]),
            nn.ReLU(),
            nn.Linear(embed_dim[0][1], num_cls)
        )

    def forward(self, similarity, caption_feats, pair_feats, label, pair_attention_label):
        # caption_feats size (batchsize, 2054)
        # pair_feats size (batchsize, 3072)
            # img_feat size (batchsize, 1536)
            # tag_feat size (batchsize, 1536)
        attended_values = self.SelfAttention(caption_feats[:, 0:2, :])
        x = attended_values.mean(dim=1)  # torch.Size([35, 2054])
        x = x * similarity  # torch.Size([35, 2054])
        x = self.projection(x)
        x = torch.cat([pair_feats, x], dim=-1)
        x = x.view(x.shape[0], 1, x.shape[1])
        x, _ = self.MLA(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        bag_logits = torch.sum(x, dim=0)
        return bag_logits

class SMART_Expert(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls):
        super(SMART_Expert, self).__init__()
        self.Expert = SMART(embed_dim, num_heads, num_cls)

    def forward(self, similarity, caption_feats, pair_feats, label, pair_attention_label):
        bag_logits = self.Expert(similarity, caption_feats, pair_feats, label, pair_attention_label)
        return bag_logits

class MyAttentionClassifier3(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cls):
        super(MyAttentionClassifier3, self).__init__()
        self.Similarity = ClipSimilarity(embed_dim[0][1] // 2)
        self.experts = nn.ModuleList([SMART_Expert(embed_dim, num_heads, num_cls) for i in range(3)])
        embed_dim = embed_dim[0][0] * 2 + embed_dim[0][1]
        self.gatting = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=1)
        )

    def forward(self, caption_feats, pair_feats, label, pair_attention_label):
        sub_img_feat = pair_feats[:, :768]
        sub_tag_feat = pair_feats[:, 768: 768 * 2]
        obj_img_feat = pair_feats[:, 768 * 2: 768 * 3]
        obj_tag_feat = pair_feats[:, 768 * 3:]
        pair_img_feat = torch.cat([sub_img_feat, obj_img_feat], dim=-1)
        similarity = self.Similarity(pair_img_feat, caption_feats)
        values, indices = similarity.squeeze().topk(1)
        caption_feat = caption_feats[indices, 0:2, :].view(1, -1)
        pair_feat = pair_feats[indices, :].view(1, -1)
        feat = torch.cat([pair_feat, caption_feat], dim=-1)
        gates = self.gatting(feat)
        expert_outs = [expert(similarity, caption_feats, pair_feats, label, pair_attention_label) for expert in
                       self.experts]
        weighted_outs = [gate * expert_out for gate, expert_out in zip(gates.unbind(1), expert_outs)]
        bag_logits = torch.stack(weighted_outs, dim=-1).sum(dim=-1).squeeze()
        if not self.training:
            bag_logits = nn.functional.softmax(bag_logits, dim=-1)
        return bag_logits, 0

