import numpy as np
from collections import defaultdict
import sklearn.metrics
import json
import pickle
import torch

def cal_metrics(sorted_results, facts):
    correct = 0
    precisions = []
    recalls = []

    num_facts_of_predicates = defaultdict(int)
    for f in facts:
        p = f[2]
        num_facts_of_predicates[p] += 1
    recalls_of_predicate = {p: [] for p in num_facts_of_predicates}
    precision_of_predicate = {p: [] for p in num_facts_of_predicates}
    f1_of_predicate = {p: [] for p in num_facts_of_predicates}
    correct_of_predicates = {p: 0 for p in num_facts_of_predicates}
    count_predication_of_p = {p: 0 for p in num_facts_of_predicates}

    class_pair_result = {}
    pr_curve_labels = []  # binary array
    pr_curve_predictions = []  # scores

    for i, item in enumerate(sorted_results):
        p = item[2]
        p_score = item[3]
        pair_key = (item[0], item[1])
        if p in num_facts_of_predicates:
            count_predication_of_p[p] += 1
        if pair_key not in class_pair_result:
            class_pair_result[pair_key] = {
                'pred': np.zeros((101), dtype=int),
                'label': np.zeros((101), dtype=int),
                'score': np.zeros((101), dtype=float)
            }

        if item[:3] in facts:
            correct += 1
            correct_of_predicates[p] += 1
            class_pair_result[pair_key]['label'][p] = 1
        class_pair_result[pair_key]['score'][p] = item[3]
        pr_curve_labels.append(item[:3] in facts)
        pr_curve_predictions.append(item[3])

        precisions.append(correct / (i + 1))
        recalls.append(correct / len(facts))

        if p in num_facts_of_predicates:
            pr = correct_of_predicates[p] / count_predication_of_p[p]
            rc = correct_of_predicates[p] / max(1, num_facts_of_predicates[p])
            recalls_of_predicate[p].append(rc)
            precision_of_predicate[p].append(pr)
            f1_of_predicate[p].append((2 * pr * rc) / (pr + rc + 1e-20))

    label_vec = []
    pred_result_vec = []
    score_vec = []
    for cls_p in class_pair_result:
        label_vec.append(class_pair_result[cls_p]['label'])
        pred_result_vec.append(class_pair_result[cls_p]['pred'])
        score_vec.append(class_pair_result[cls_p]['score'])
    label_vec = np.stack(label_vec, 0)
    pred_result_vec = np.stack(pred_result_vec, 0)
    score_vec = np.stack(score_vec, 0)

    np_recall_of_predicates = {p: np.array(r) for p, r in recalls_of_predicate.items()}
    np_precision_of_predicate = {p: np.array(pr) for p, pr in precision_of_predicate.items()}
    max_f1_of_predicate = {p: np.array(f1).max() for p, f1 in f1_of_predicate.items()}
    auc_of_predicate = {p: sklearn.metrics.auc(x=np_recall_of_predicates[p], y=np_precision_of_predicate[p]) for p, f1
                        in f1_of_predicate.items()}

    auc = sklearn.metrics.auc(x=recalls, y=precisions)
    np_prec = np.array(precisions)
    np_rec = np.array(recalls)
    max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()

    best_threshold = sorted_results[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()][3].item()

    pred_result_vec = score_vec >= best_threshold
    valid_p = list(num_facts_of_predicates.keys())
    # max_macro_f1 = sklearn.metrics.f1_score(label_vec[:, valid_p],
    #                                         pred_result_vec[:, valid_p],
    #                                         average='macro')
    max_macro_f1 = sum(max_f1_of_predicate.values()) / len(max_f1_of_predicate)
    assert len(auc_of_predicate) == len(valid_p)
    macro_auc = sum(auc_of_predicate.values()) / len(auc_of_predicate)
    macro_p = sum(np_precision_of_predicate.values(), np.zeros(len(np_recall_of_predicates[40]))) / len(
        np_precision_of_predicate)
    return label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, max_macro_f1, np.array(
        pr_curve_labels), np.array(pr_curve_predictions)

def load_cached_image_ids(data_dir, split):
    idx_file_name = f'{data_dir}/{split}_image_sort_by_IoU_and_distance.pkl'
    idx_to_id_fname = f'{data_dir}/{split}_image_idx_to_id.pkl'

    idx_to_id = pickle.load(open(idx_to_id_fname, 'rb'))
    bag_key_to_idx = pickle.load(open(idx_file_name, 'rb'))
    mapping = json.load(open(f'{data_dir}/vg_dict.json'))

    _150_idx_to_label = {v: k for k, v in mapping['label_to_vg_150_idx'].items()}
    bag_key_to_ids = {
        '#'.join([str(mapping['label_to_idx'][_150_idx_to_label[x]]) for x in k]): [idx_to_id[x] for x in v] for k, v in
        bag_key_to_idx.items()}

    return bag_key_to_ids

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_vg_150_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        # info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_vg_150_idx']
    predicate_to_ind = info['predicate_to_idx']
    # attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    # ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])
    _100_cls_idx_to_vg150_idx = {int(k): info['label_to_vg_150_idx'][info['idx_to_label'][k]] for k in
                                 info['idx_to_label']}
    return ind_to_classes, ind_to_predicates, [], _100_cls_idx_to_vg150_idx

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def process_array(tokens_a):
    arr = tokens_a.numpy().tolist()[0]
    end = len(arr)
    while end > 0 and arr[end - 1] == 0:
        end -= 1
    trimmed = arr[:end]

    if len(trimmed) >= 2:
        return trimmed[1:-1]
    else:
        return []

# Bert
def tokenize(tokenizer, text_a, text_b, img_feat, max_img_seq_len=50,
             max_seq_a_len=40, max_seq_len=70, cls_token_segment_id=0,
             pad_token_segment_id=0, sequence_a_segment_id=0, sequence_b_segment_id=1):
    tokens_a = tokenizer.tokenize(text_a)
    # print("text_a: ", text_a)
    # print("tokens_a: ", tokens_a)
    # print("len(tokens_a): ", len(tokens_a))
    # print("type(tokens_a): ", [type(token_a) for token_a in tokens_a])
    # tokens_b = None # deprecated
    # if text_b:
    #     # deprecated
    #     tokens_b = tokenizer.tokenize(text_b)
    #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
    # else:
    #     if len(tokens_a) > max_seq_len - 2:
    #         tokens_a = tokens_a[:(max_seq_len - 2)]
    if len(tokens_a) > max_seq_len - 2:
        tokens_a = tokens_a[:(max_seq_len - 2)]
    t1_label = len(tokens_a) * [-1]
    # if tokens_b:
    #     t2_label = [-1] * len(tokens_b)
    lm_label_ids = ([-1] + t1_label + [-1])
    # concatenate lm labels and account for CLS, SEP, SEP
    # if tokens_b:
    #     lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    # else:
    #     lm_label_ids = ([-1] + t1_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = [] # deprecated
    tokens.append("[CLS]")
    segment_ids.append(0) # deprecated
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0) # deprecated
    tokens.append("[SEP]")
    segment_ids.append(0) # deprecated

    # deprecated
    # if tokens_b:
    #     assert len(tokens_b) > 0
    #     for token in tokens_b:
    #         tokens.append(token)
    #         segment_ids.append(1)
    #     tokens.append("[SEP]")
    #     segment_ids.append(1)
    # print("tokens: ", tokens)
    # print("len(tokens): ", len(tokens))
    # print("type(tokens): ", [type(token) for token in tokens])
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    # print("input_ids: ", input_ids)
    # print("len(input_ids): ", len(input_ids))
    # print("type(input_ids): ", [type(input_id) for input_id in input_ids])
    # exit(110)
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    # image features
    if max_img_seq_len > 0:
        img_feat_len = img_feat.shape[0]
        if img_feat_len > max_img_seq_len:
            input_mask = input_mask + [1] * img_feat_len
        else:
            input_mask = input_mask + [1] * img_feat_len
            pad_img_feat_len = max_img_seq_len - img_feat_len
            input_mask = input_mask + ([0] * pad_img_feat_len)

    lm_label_ids = lm_label_ids + [-1] * max_img_seq_len

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
    return input_ids, input_mask, segment_ids, tokens_a, lm_label_ids