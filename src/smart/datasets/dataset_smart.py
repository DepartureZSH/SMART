import json
import random
import base64

import numpy as np
import torch
import torch.distributed
from torch.utils.data import Dataset
from smart.utils.tsv_file import TSVFile
from smart.datasets import real_bag_size, feature_folder
from smart.datasets.dataset_utils import tokenize, load_cached_image_ids, cal_metrics

class BagDatasetPairAsUnit(Dataset):
    def __init__(self, data_dir, bag_data_file, split, args=None, tokenizer=None, txt_seq_len=70, img_seq_len=50,
                 shuffle=False, **kwargs):
        self.split = split

        # self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.txt_seq_len = txt_seq_len
        self.img_seq_len = img_seq_len

        self.bag_data = json.load(open(bag_data_file))
        self.idx_to_bag_key = {i: k for i, k in enumerate(self.bag_data.keys())}

        self.label_tsv = TSVFile(f'{data_dir}/VG_100_100_label.tsv')
        self.line_tsv = TSVFile(f'{data_dir}/{split}_feat_idx_to_label_line.tsv')
        self.prediction = TSVFile(f'{data_dir}/obj_feat_{split}.tsv')
        ########################################################################################
        # Features
        # Blip1 object(local) caption without prompt
        self.captions_local_v1 = TSVFile(f'{data_dir}/{feature_folder}/BLIP1/obj_feat_{split}.tsv')
        # Blip2 object(local) caption without prompt
        self.captions_local_v2 = TSVFile(f'{data_dir}/{feature_folder}/BLIP2/Local/obj_feat_{split}.tsv')
        # Blip2 image(Global) caption without prompt
        self.captions_global_v2 = TSVFile(f'{data_dir}/{feature_folder}/BLIP2/Global/obj_feat_{split}.tsv')
        ########################################################################################

        self.idx_to_image_id = json.load(open(f'{data_dir}/Features/idx_image_id_mapping.json'))
        self.bag_pair_data = json.load(open(f'{data_dir}/{split}_pairs_data.json'))
        self.img_id_to_key = {k: v[0] for k, v in enumerate(self.label_tsv)}

        self.key_to_prediction_line = {v[0]: i for i, v in enumerate(self.prediction)}

        self.predicate_to_idx = json.load(open(f'{data_dir}/vg_dict.json'))['predicate_to_idx']
        self.shuffle = shuffle

        mapping = json.load(open(f'{data_dir}/vg_dict.json'))
        self.idx_to_cls_name = mapping['idx_to_label']

        self.facts = set()
        for key in self.bag_data:
            label = self.bag_data[key]['label']
            key = key.split('#')
            if args.num_classes == 100:
                for l in label:
                    if l:
                        self.facts.add((key[0], key[1], l))
            elif args.num_classes == 101:
                for l in label:
                    self.facts.add((key[0], key[1], l))
        print(f'{len(self.facts)} facts for {bag_data_file}')

        self.bag_key_to_sorted_image_ids = load_cached_image_ids(data_dir, split)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return len(self.idx_to_bag_key)

    def get_image_item(self, item, pairs):
        # load data
        img_108076_id_str, object_classes, object_captions, object_features, object_boxes, image_file, label_captions_tokens = self.decode_features(item)
        assert len(object_features) == len(object_boxes)
        

        object_classes = object_classes[: self.img_seq_len]
        if len(object_captions) == 2:
            image_caption_v2 = object_captions[1]
            object_captions_v2 = object_captions[0][: self.img_seq_len]
            object_captions = label_captions_tokens
            object_cap_feat_v2_tokens = [self.caption_tokenizer(caption['caption']) for caption in object_captions_v2]
            image_cap_feat_v2_tokens = self.caption_tokenizer(image_caption_v2[0]["G_caption"])
            image_cap_feats_v2 = image_cap_feat_v2_tokens.view(1, -1)
            # print(len(object_cap_feat_v2_tokens), ",", object_cap_feat_v2_tokens[0].shape)
            # print(image_cap_feat_v2_tokens.shape)
        else:
            object_captions = object_captions[: self.img_seq_len]
        object_features = object_features[: self.img_seq_len]
        object_boxes = object_boxes[: self.img_seq_len]
        pairs = [p for p in pairs if p[0] < self.img_seq_len and p[1] < self.img_seq_len]

        pair_image_feats = []
        pair_cap_feats = []
        pair_cap_feats_v2 = []
        pair_input_ids = []
        pair_input_mask = []
        pair_segment_ids = []
        pair_object_boxes = []
        pair_object_classes = []
        pair_image_cap_feats_v2 = []
        pair_object_name_positions = []
        for pair in pairs:
            obj_idx_permutation = [pair[0], pair[1],
                                   *[idx for idx in range(len(object_boxes)) if idx not in pair]]
            obj_class_permutation = [object_classes[idx] for idx in obj_idx_permutation]

            object_tag_text = " ".join(obj_class_permutation)
            text_a = object_tag_text
            text_b = ''

            # generate features
            input_ids, input_mask, segment_ids, tokens_a, _ = tokenize(self.tokenizer,
                                                                        text_a=text_a, text_b=text_b,
                                                                        img_feat=object_features,
                                                                        max_img_seq_len=self.img_seq_len,
                                                                        max_seq_a_len=70, max_seq_len=70,
                                                                        cls_token_segment_id=0,
                                                                        pad_token_segment_id=0,
                                                                        sequence_a_segment_id=0,
                                                                        sequence_b_segment_id=1)
            object_name_positions = []
            current_object_positions = []
            for token_idx, tok in enumerate(tokens_a, 1):  # omit [CLS]
                tok: str

                # find a new name, save word-piece positions of previous one
                if not tok.startswith('##'):
                    object_name_positions.append(current_object_positions)
                    current_object_positions = []
                current_object_positions.append(token_idx)
            del object_name_positions[0]
            object_name_positions.append(current_object_positions)

            object_num = object_features.size(0)
            # print("Object_features size ", object_features.size())
            img_feat = torch.cat([torch.stack([object_features[idx] for idx in obj_idx_permutation]),
                                  torch.zeros([self.img_seq_len - object_num, 2054])], 0)

            cap_feat = torch.cat([torch.stack([object_captions[idx] for idx in obj_idx_permutation]),
                                  torch.zeros([self.img_seq_len - object_num, 2054])], 0)

            # image_cap_feat_v2 = 
            object_cap_feats_v2 = torch.cat([torch.stack([object_cap_feat_v2_tokens[idx] for idx in obj_idx_permutation]), torch.zeros([self.img_seq_len - object_num, 768])], 0)
            
            assert len(object_name_positions) == len(object_classes)
            pair_image_feats.append(img_feat)
            pair_cap_feats.append(cap_feat)
            pair_cap_feats_v2.append(object_cap_feats_v2)
            pair_input_ids.append(input_ids)
            pair_input_mask.append(input_mask)
            pair_segment_ids.append(segment_ids)
            pair_object_boxes.append([object_boxes[idx] for idx in obj_idx_permutation])
            pair_object_classes.append(obj_class_permutation)
            pair_image_cap_feats_v2.append(image_cap_feats_v2)
            pair_object_name_positions.append(object_name_positions)

        return (img_108076_id_str, pair_image_feats, pair_cap_feats, pair_input_ids, pair_input_mask, pair_segment_ids,
                pair_object_boxes, pair_object_classes, pair_object_name_positions, image_file, pair_cap_feats_v2, pair_image_cap_feats_v2)

    def __getitem__(self, item):
        bag_key = self.idx_to_bag_key[item]
        bag_data = self.bag_data[bag_key]
        bag_pair_data = self.bag_pair_data[bag_key]
        bag_label = torch.tensor(random.choice(bag_data['label']))

        bag_image_ids = self.bag_key_to_sorted_image_ids[bag_key][:real_bag_size]
        rel_bag_image_ids = bag_data['relation_image_ids']

        bag_pairs = bag_pair_data[:real_bag_size]
        # {"image_id": (subject1_idx, object1_idx),(subject2_idx, object2_idx)...}
        bag_image_id_to_pairs = {}
        attention_label = [int(img_id in rel_bag_image_ids) for img_id, *_ in bag_pairs]
        for pair in bag_pairs:
            img_id, sub_idx, obj_idx, v = pair
            if img_id not in bag_image_id_to_pairs:
                bag_image_id_to_pairs[img_id] = []
            bag_image_id_to_pairs[img_id].append((sub_idx, obj_idx))

        bag_image_feats = []
        bag_caption_feats = []
        bag_caption_feats_v2 = []
        bag_input_ids = []
        bag_input_mask = []
        bag_segment_ids = []
        bag_object_boxes = []
        bag_object_classes = []
        bag_object_name_positions = []
        bag_image_files = []
        bag_image_cap_feats_v2 = []
        for img_id, pairs in bag_image_id_to_pairs.items():
            idx = self.key_to_prediction_line[self.img_id_to_key[img_id]]
            (img_108076_id_str, pair_image_feats, pair_cap_feats, pair_input_ids, pair_input_mask, pair_segment_ids,
             pair_object_boxes, pair_object_classes, pair_object_name_positions, image_file, pair_cap_feats_v2, image_cap_feats_v2) = self.get_image_item(idx, pairs)
            bag_image_feats += pair_image_feats
            bag_caption_feats += pair_cap_feats
            bag_caption_feats_v2 += pair_cap_feats_v2
            bag_input_ids += pair_input_ids
            bag_input_mask += pair_input_mask
            bag_segment_ids += pair_segment_ids
            bag_object_boxes += pair_object_boxes
            bag_object_classes += pair_object_classes
            bag_object_name_positions += pair_object_name_positions
            bag_image_files.append(image_file)
            bag_image_cap_feats_v2 += image_cap_feats_v2

        bag_image_feats = torch.stack(bag_image_feats, 0)
        bag_caption_feats = torch.stack(bag_caption_feats, 0)
        bag_caption_feats_v2 = torch.stack(bag_caption_feats_v2, 0)
        bag_image_cap_feats_v2 = torch.stack(bag_image_cap_feats_v2, 0)
        bag_input_ids = torch.stack(bag_input_ids, 0)
        bag_input_mask = torch.stack(bag_input_mask, 0)
        bag_segment_ids = torch.stack(bag_segment_ids, 0)

        return (bag_key, bag_image_feats, bag_caption_feats, bag_input_ids, bag_input_mask, bag_segment_ids, bag_object_boxes,
                bag_object_classes, bag_object_name_positions, [], [], bag_label, attention_label, bag_image_ids, bag_image_files, bag_caption_feats_v2, bag_image_cap_feats_v2)

    def eval(self, pred_result):
        seen = set()
        deduplicated_pred_result = []
        for item in pred_result:
            if (item['class_pair'][0], item['class_pair'][1], item['relation']) in seen:
                continue
            deduplicated_pred_result.append(item)
            seen.add((item['class_pair'][0], item['class_pair'][1], item['relation']))
        pred_result = deduplicated_pred_result

        # print(f'Sorting evaluation results, size={len(pred_result)}')
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)

        sorted_pred_result = [(x['class_pair'][0], x['class_pair'][1], x['relation'], x['score']) for x in
                              sorted_pred_result]

        label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, \
        max_macro_f1, pr_curve_labels, pr_curve_predictions = cal_metrics(sorted_pred_result, self.facts)
        return {
            'results': sorted_pred_result,
            'auc': auc,
            'macro_auc': macro_auc,
            'max_micro_f1': max_micro_f1,
            'max_macro_f1': max_macro_f1,
            'p@2%': np_prec[int(0.02 * len(np_prec))],
            'mp@2%': macro_p[int(0.02 * len(macro_p))],
            'recalls': np_rec.tolist(),
            'pr_curve_labels': pr_curve_labels,
            'pr_curve_predictions': pr_curve_predictions
        }

    def decode_features(self, item_idx):
        img_108076_id_str, prediction_str = self.prediction.seek(item_idx)
        # img_108076_id_str2, prediction_str2 = self.captions.seek(item_idx)
        # img_108076_id_str3, prediction_str3 = self.obj_captions.seek(item_idx)
        _, caption_str = self.captions_local_v1.seek(item_idx)
        _, caption_str1 = self.captions_local_v2.seek(item_idx)
        _, caption_str2 = self.captions_global_v2.seek(item_idx)
        image_file = self.idx_to_image_id[img_108076_id_str]
        
        assert img_108076_id_str == _
        # print(f"prediction_str2 {prediction_str2}")
        # print(f"prediction_str3 {prediction_str3}")
        # print(self.idx_to_image_id[img_108076_id_str3])
        # print(f"caption_str {json.loads(caption_str)["objects"][0]}")
        # exit(1)
        # print(f"prediction_str {prediction_str}")
        feat_info = json.loads(prediction_str)
        caption_info = json.loads(caption_str)
        caption_info1 = json.loads(caption_str1)
        caption_info2 = json.loads(caption_str2)

        label_tsv_row_idx = int(self.line_tsv[item_idx][0])
        _, annotation_str = self.label_tsv[label_tsv_row_idx]
        assert _ == img_108076_id_str

        annotation_info = json.loads(annotation_str)
        objects_annotation, relationships_annotation = annotation_info['objects'], annotation_info['relations']

        # prediction_objects: [{'rect':, 'feature':}, ...]
        prediction_objects = feat_info["objects"]
        object_features = [np.frombuffer(base64.b64decode(o['feature']), np.float32) for o in prediction_objects]
        object_features = torch.Tensor(np.stack(object_features))

        # class names
        prediction_classes = [o['class'] for o in prediction_objects]
        label_classes = [o['class'] for o in objects_annotation]
        assert len(label_classes) == len(prediction_classes)
        
        # object captions
        object_captions  = caption_info["objects"]
        object_captions1 = caption_info1["objects"]
        object_captions2 = caption_info2["objects"]
        label_captions = [object_captions1, object_captions2]
        captions_features = [np.frombuffer(base64.b64decode(o['caption_token']), np.float32) for o in object_captions]
        label_captions_tokens  = torch.Tensor(np.stack(captions_features))


        # bboxes
        prediction_boxes = [o['rect'] for o in prediction_objects]
        label_boxes = [o['rect'] for o in objects_annotation]
        assert len(prediction_boxes) == len(label_boxes)

        return img_108076_id_str, label_classes, label_captions, object_features, label_boxes, image_file, label_captions_tokens

    def caption_tokenizer(self, text):
        tokens = self.tokenizer.tokenize(text)
        caption_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        caption_ids = torch.nn.functional.pad(torch.tensor(caption_ids), (0, 768 - len(caption_ids)), value = 0)
        return caption_ids

class args:
    num_classes = 100
    eval_model_dir = 'bert-base-uncased'
    model = "SMART"
    import pathlib
    folder = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
    train_dir = f'{folder}/data/'
    per_gpu_eval_batch_size = 1
    num_workers = 1
    device = "cuda"
    def __init__(self):
        pass

if __name__ == "__main__":
    import pathlib
    folder = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
    from smart.modeling.pytorch_transformers import BertTokenizer
    ######################################################################################################
    # Load pretrained model and tokenizer
    tokenizer_class = BertTokenizer
    checkpoint = f"{folder}/pretrained_models/pretrained_base/"
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    from smart.utils.initizer import make_data_loader
    split = 'test'
    dataloader = make_data_loader(
        args, None, f'{folder}/data/{split}_bag_data.json', 'test', tokenizer,
        is_distributed=False,
        is_train=False)
    # Dataset = BagDatasetPairAsUnit(data_dir=f'{folder}/data', bag_data_file=f'{folder}/data/{split}_bag_data.json', split=split, args=args, tokenizer=tokenizer, txt_seq_len=70, img_seq_len=50, shuffle=False)
    # if Dataset[252][0] == '2#9':
    #     print(Dataset[252])
    for step, (label_list, batch) in enumerate(dataloader):
        (bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
             bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list,
             bag_image_ids_list, bag_image_files, bag_caption_feats_v2, bag_image_cap_feats_v2) = label_list
        batch = tuple([x.to(args.device) for x in t] for t in batch)
        img_feat, caption_feat, input_ids, input_mask, segment_ids = batch
        print(bag_caption_feats_v2[0].shape)
        print(bag_image_cap_feats_v2[0].shape)
        print(caption_feat[0].shape)
        print(torch.cat([bag_image_cap_feats_v2[0], bag_caption_feats_v2[0]], dim=1).shape)
        break
    