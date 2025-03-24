import pickle
import pathlib
import json
import numpy as np
import sklearn
from collections import defaultdict
import sklearn.metrics
from sklearn.metrics import PrecisionRecallDisplay
from utils.tsv_file import TSVFile
folder = pathlib.Path(__file__).parent.parent.parent.resolve()
        # real_rels = bag_data[f"{sub}#{obj}"]['label']
        # print(f"({sub}, {obj}, {rel}, {conf})-{real_rels}")
        # print(f"({i2l[sub]}, {i2l[obj]}, {i2p[str(rel)]}, {conf})-{[i2p[str(real_rel)] for real_rel in real_rels]}")
from PIL import Image
import matplotlib.pyplot as plt


def show_img(path: str, ) -> None:
    img = Image.open(fp=path)
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(img)  # 将数据显示为图像，即在二维常规光栅上。
    plt.show()  # 显示图片

class analsyser:
    def __init__(self, folder, path):
        test_bag_file = f"{folder}/data/test_bag_data.json"
        self.bag_data = json.load(open(test_bag_file))
        self.idx_to_image_id = json.load(open(f"{folder}/data/Features/idx_image_id_mapping.json"))
        self.facts = set()
        n = 0
        for key in self.bag_data:
            label = self.bag_data[key]['label']
            key = key.split('#')
            for l in label:
                if path != "":
                    self.facts.add((key[0], key[1], l))
                elif l:
                    self.facts.add((key[0], key[1], l))
        # print(n)
        # print(len(facts))
        vg_dict_file = f"{folder}/data/vg_dict.json"
        vg_dict = json.load(open(vg_dict_file))
        self.i2l = vg_dict["idx_to_label"]
        self.i2p = vg_dict["idx_to_predicate"]
        self.i2p.update({"0": "no realtion"})
        self.results = pickle.load(open(f"{folder}/output/Analysis/{path}/best_results.pkl", 'rb'))
        self.results_map = {}
        for each in self.results:
            if self.results_map.get(f"{each[0]}#{each[1]}"):
                self.results_map[f"{each[0]}#{each[1]}"].append([self.i2p[str(each[2])], each[3]])
            else:
                self.results_map[f"{each[0]}#{each[1]}"]=[[self.i2p[str(each[2])], each[3]]]

        self.idx_to_bag_key = {i: k for i, k in enumerate(self.bag_data.keys())}
        self.prediction = TSVFile(f'{folder}/data/obj_feat_test.tsv')
        self.label_tsv = TSVFile(f'{folder}/data/VG_100_100_label.tsv')
        self.key_to_prediction_line = {v[0]: i for i, v in enumerate(self.prediction)}
        self.img_id_to_key = {k: v[0] for k, v in enumerate(self.label_tsv)}
        self.bag_pair_data = json.load(open(f'{folder}/data/test_pairs_data.json'))
        rels = set()
        y = []
        scores = []
        for sub, obj, rel, conf in self.results:
            if (sub, obj, rel) in self.facts:
                y.append(rel)
                scores.append(conf)
            if self.bag_data[f"{sub}#{obj}"]['label'] == [0]:
                rels.add((sub, obj, rel, conf))

    def cal_metrics(self):
        sorted_results = self.results
        facts = self.facts
        correct = 0
        precisions = []
        recalls = []

        # relation label 数量
        num_facts_of_predicates = defaultdict(int)
        for f in facts:
            p = f[2]
            num_facts_of_predicates[p] += 1

        recalls_of_predicate = {p: [] for p in num_facts_of_predicates}
        precision_of_predicate = {p: [] for p in num_facts_of_predicates}
        f1_of_predicate = {p: [] for p in num_facts_of_predicates}
        correct_of_predicates = {p: 0 for p in num_facts_of_predicates}
        count_predication_of_p = {p: 0 for p in num_facts_of_predicates}
        # print(len(num_facts_of_predicates))

        class_pair_result = {}
        pr_curve_labels = []  # binary array
        pr_curve_predictions = []  # scores

        # print(sorted_results)
        p_scores = 0
        for i, item in enumerate(sorted_results):
            p = item[2]
            p_score = item[3]
            pair_key = (item[0], item[1])
            # if pair_key == ('27', '16'):
            #     p_scores += p_score
            #     print(f"Pair Key: {pair_key}, Predicate: {p} {self.i2p[str(p)]}, Score: {p_score}, Is Correct: {item[:3] in facts}")
            #     print(f"sum score {p_scores}")
            # count_predication_of_p 记录预测结果 在 label集的数量
            # TP + FP
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
                # TP
                correct_of_predicates[p] += 1
                class_pair_result[pair_key]['label'][p] = 1
            # if p == 12:
            #     print(item[:4])
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
        self.np_recall_of_predicates = np_recall_of_predicates
        np_precision_of_predicate = {p: np.array(pr) for p, pr in precision_of_predicate.items()}
        self.np_precision_of_predicate = np_precision_of_predicate
        max_f1_of_predicate = {p: np.array(f1).max() for p, f1 in f1_of_predicate.items()}
        # rc1 = np_recall_of_predicates[12]
        # pr1 = np_precision_of_predicate[12]
        # print(np_recall_of_predicates[12])
        # print(np_precision_of_predicate[12])
        auc_of_predicate = {
            p: sklearn.metrics.auc(x=np_recall_of_predicates[p],y=np_precision_of_predicate[p])
            for p, f1 in f1_of_predicate.items()}
        auc_of_predicate1 = {p: 1 if all(np_recall_of_predicates[p]==1) else sklearn.metrics.auc(x=np_recall_of_predicates[p], y=np_precision_of_predicate[p])
                            for p, f1 in f1_of_predicate.items()}

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
        self.my_m1 = [(str(key), conf) for key, conf in sorted(auc_of_predicate.items(), key=lambda x: x[1], reverse=False)]
        # print([(i2p[str(key)], conf) for key, conf in sorted(auc_of_predicate.items(), key=lambda x: x[1], reverse=False)])
        macro_auc = sum(auc_of_predicate.values()) / len(auc_of_predicate)
        macro_auc1 = sum(auc_of_predicate1.values()) / len(auc_of_predicate1)
        macro_p = sum(np_precision_of_predicate.values(), np.zeros(len(np_recall_of_predicates[40]))) / len(
            np_precision_of_predicate)
        return label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, [macro_auc, macro_auc1], auc, max_micro_f1, max_macro_f1, np.array(
            pr_curve_labels), np.array(pr_curve_predictions)

    def show_bugs(self, n):
        # print("AUC per class: ", self.my_m1)
        # print("AUC per class: ", [(self.i2p[key], conf) for key, conf in self.my_m1])
        notes = {i: [] for i in range(0, n)}
        for fact in list(self.facts):
            for i in range(0, n):
                if str(fact[2]) == self.my_m1[i][0]:
                    notes[i].append(fact)
                    # print(fact)
                    # print(self.bag_data[f"{fact[0]}#{fact[1]}"])
        print({self.i2p[self.my_m1[key][0]]: [f"{note[0]}#{note[1]}" for note in notes[key]] for key in notes.keys()})
        return {self.i2p[self.my_m1[key][0]]: [f"{note[0]}#{note[1]}" for note in notes[key]] for key in notes.keys()}

    def show_bag(self, bag_key):
        # bag_key = self.idx_to_bag_key[item]
        bag_data = self.bag_data[bag_key]
        bag_pair_data = self.bag_pair_data[bag_key]
        bag_image_id_to_pairs = {}
        bag_pairs = bag_pair_data[:50]
        for pair in bag_pairs:
            img_id, sub_idx, obj_idx, v = pair
            if img_id not in bag_image_id_to_pairs:
                bag_image_id_to_pairs[img_id] = []
            bag_image_id_to_pairs[img_id].append((sub_idx, obj_idx))
        for img_id, pairs in bag_image_id_to_pairs.items():
            idx = self.key_to_prediction_line[self.img_id_to_key[img_id]]
            img_108076_id_str, prediction_str = self.prediction.seek(idx)
            image_file = self.idx_to_image_id[img_108076_id_str]
            path = f"{folder}/data/newdata/{image_file}"
            show_img(path)

if __name__ == '__main__':
    anal = analsyser(folder, "MoE")
    label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, \
        max_macro_f1, pr_curve_labels, pr_curve_predictions = anal.cal_metrics()
    metrics = {
        'auc': auc,
        'macro_auc': macro_auc[0],
        'macro_auc1': macro_auc[1],
        'max_micro_f1': max_micro_f1,
        'max_macro_f1': max_macro_f1,
        'p@2%': np_prec[int(0.02 * len(np_prec))],
        'mp@2%': macro_p[int(0.02 * len(macro_p))]
    }
    print("auc ", metrics["auc"])
    print("max_micro_f1 ", metrics["max_micro_f1"])
    print("p@2% ", metrics["p@2%"])
    print("macro_auc ", metrics["macro_auc"])
    print("max_macro_f1 ", metrics["max_macro_f1"])
    print("mp@2% ", metrics["mp@2%"])
    
    print("\nmacro_auc1 ", metrics["macro_auc1"])
    # print(anal.np_precision_of_predicate)
    # results = anal.show_bugs(94)
    # labels = list(results.keys())
    # bag_keys = list(results.values())
    # for i in range(2):
    #     rank = i
    #     for k in range(len(bag_keys[rank])):
    #         sub = bag_keys[rank][k].split('#')[0]
    #         obj = bag_keys[rank][k].split('#')[1]
    #         # print(anal.results_map[bag_keys[rank][k]])
    #         print(anal.i2l[sub], labels[rank], anal.i2l[obj])
    #         for each in anal.results_map[bag_keys[rank][k]]:
    #             if labels[rank] == each[0]:
    #                 print(each)
    # anal.show_bag(bag_keys[rank][k])
    # print([(i2p[key], conf) for key, conf in my_m1])
    # for fact in list(facts):
    #     if str(fact[2]) == my_m1[1][0]:
    #         print(fact)
    #         print(bag_data[f"{fact[0]}#{fact[1]}"])
# import numpy as np
# from sklearn import metrics
# from matplotlib import pyplot as plt
# y = np.array(y)
# scores = np.array(scores)
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# plt.plot(fpr,tpr,marker = 'o')
#
# plt.show()
#
# plt.savefig(f"{folder}/output/Analysis/roc_curve.png")