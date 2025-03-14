import pickle
import pathlib
import json

from smart.datasets.dataset_utils import tokenize, load_cached_image_ids, cal_metrics
folder = pathlib.Path(__file__).parent.parent.parent.resolve()

test_bag_file = f"{folder}/data/test_bag_data.json"
bag_data = json.load(open(test_bag_file))
facts = set()
n = 0
for key in bag_data:
    label = bag_data[key]['label']
    key = key.split('#')
    for l in label:
        if l:
            facts.add((key[0], key[1], l))
# print(n)
# print(len(facts))
vg_dict_file = f"{folder}/data/vg_dict.json"
vg_dict = json.load(open(vg_dict_file))
i2l = vg_dict["idx_to_label"]
i2p = vg_dict["idx_to_predicate"]
i2p.update({"0": "no realtion"})

results = pickle.load(open(f"{folder}/output/Analysis/best_results.pkl", 'rb'))

rels = set()
y = []
scores = []
for sub, obj , rel, conf in results:
    if (sub, obj, rel) in facts:
        y.append(rel)
        scores.append(conf)
    if bag_data[f"{sub}#{obj}"]['label'] == [0]:
        rels.add((sub, obj, rel, conf))
        # real_rels = bag_data[f"{sub}#{obj}"]['label']
        # print(f"({sub}, {obj}, {rel}, {conf})-{real_rels}")
        # print(f"({i2l[sub]}, {i2l[obj]}, {i2p[str(rel)]}, {conf})-{[i2p[str(real_rel)] for real_rel in real_rels]}")
print(len(rels))
print(len(results))
# label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, \
#         max_macro_f1, pr_curve_labels, pr_curve_predictions = cal_metrics(results, facts)
# metrics = {
#     'auc': auc,
#     'macro_auc': macro_auc,
#     'max_micro_f1': max_micro_f1,
#     'max_macro_f1': max_macro_f1,
#     'p@2%': np_prec[int(0.02 * len(np_prec))],
#     'mp@2%': macro_p[int(0.02 * len(macro_p))],
#     # 'recalls': np_rec.tolist(),
#     # 'pr_curve_labels': pr_curve_labels,
#     # 'pr_curve_predictions': pr_curve_predictions
# }
# print(metrics)

import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
y = np.array(y)
scores = np.array(scores)
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
plt.plot(fpr,tpr,marker = 'o')

plt.show()

plt.savefig(f"{folder}/output/Analysis/roc_curve.png")