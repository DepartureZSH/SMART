import argparse
import os
import os.path as op
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing

from smart.utils.comm import all_gather, gather_on_master, reduce_dict, get_world_size, get_rank, is_main_process

####################################################################
# Get args from command line
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='SMART', type=str, required=True, 
                        help="The model for training.")
    parser.add_argument("--num_classes", default=100, type=int, required=False, 
                        help="How many labels in the model")
    parser.add_argument("--train_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--test_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False,
                        help="yaml file used for validation during training.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int,
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--tie_weights", default=False, action='store_true',
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=False, action='store_true',
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int,
                        help=".")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or cosine")
    parser.add_argument("--num_workers", default=3, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=40, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=5,
                        help="number of sampled captions for sc training")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--sc_beam_size', type=int, default=1,
                        help="beam size for scst training")
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p',
                        help="path to cached cPickle file used to calculate CIDEr scores")
    # for generation
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=20,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    parser.add_argument('--sfmx_t', type=float, default=18)
    parser.add_argument('--loss_w_t', type=float, default=-1.0)
    parser.add_argument('--attention_w', type=float, default=0.3)
    parser.add_argument('--head', type=str, default='att')
    parser.add_argument('--select_size', type=int, default=10)
    parser.add_argument('--real_bag_size', type=int, default=10)
    parser.add_argument('--mAUC_weight', type=int, default=1)
    parser.add_argument('--pretrained_weight', type=str, default='')
    parser.add_argument('--VRD_weight', type=str, default='')

    args = parser.parse_args()

    return args

####################################################################
# Hardware Initializer
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Setup CUDA, GPU & distributed training
def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank

# Setup CUDA, GPU & distributed training
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

####################################################################
# Save model settings
def restore_training_settings(args, logger):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
            max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
                       'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                                                                              test_v, train_v))
                setattr(args, param, train_v)
    return args

####################################################################
# Data Loader Functions
# Load {split} data
def make_data_loader(args, logger, bag_data_file, split, tokenizer, is_distributed=True, is_train=False):
    if is_train:
        dataset = build_train_dataset(bag_data_file, split, tokenizer, args)
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
        if args.model == "CLEVER":
            collate_fn = collate
        elif args.model == "SMART":
            collate_fn = collate_smart
    else:
        dataset = build_test_dataset(bag_data_file, split, tokenizer, args)
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        if args.model == "CLEVER":
            collate_fn = collate
        elif args.model == "SMART":
            collate_fn = collate_smart

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return data_loader

def collate(batch):
    new_batch = list(zip(*batch))
    (bag_key_list, bag_image_feats_list, bag_input_ids_list, bag_input_mask_list, bag_segment_ids_list,
     bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
     bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_label_list,
     attention_label_list, bag_image_ids_list, preload_ids_list) = new_batch

    bag_labels = torch.stack(bag_label_list, 0)
    return ((bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
             bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list, bag_image_ids_list,
             preload_ids_list),
            (bag_image_feats_list, bag_input_ids_list, bag_input_mask_list, bag_segment_ids_list))

def collate_smart(batch):
    new_batch = list(zip(*batch))
    (bag_key_list, bag_image_feats_list, bag_caption_feats_list, bag_input_ids_list, bag_input_mask_list, bag_segment_ids_list,
     bag_object_boxes_list,
     bag_object_classes_list, bag_object_name_positions_list,
     bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_label_list,
     attention_label_list, bag_image_ids_list, preload_ids_list) = new_batch

    bag_labels = torch.stack(bag_label_list, 0)
    return ((bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
             bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list, bag_image_ids_list,
             preload_ids_list),
            (bag_image_feats_list, bag_caption_feats_list, bag_input_ids_list, bag_input_mask_list, bag_segment_ids_list))

def build_test_dataset(bag_data_file, split, tokenizer, args):
    if args.model == "CLEVER":
        from smart.datasets.dataset_clever import BagDatasetPairAsUnit
    elif args.model == "SMART":
        from smart.datasets.dataset_smart import BagDatasetPairAsUnit
    return BagDatasetPairAsUnit(args.train_dir, bag_data_file, split, tokenizer=tokenizer, args=args, shuffle=False)

def build_train_dataset(bag_data_file, split, tokenizer, args):
    if args.model == "CLEVER":
        from smart.datasets.dataset_clever import BagDatasetPairAsUnit
    elif args.model == "SMART":
        from smart.datasets.dataset_smart import BagDatasetPairAsUnit
    return BagDatasetPairAsUnit(args.train_dir, bag_data_file, split, tokenizer=tokenizer, args=args, shuffle=False)

## Dataloader Make data sampler
def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, rank=get_rank(),
                                                               num_replicas=get_world_size())
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

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

####################################################################
# Tensorboard Writer
## Write scalars
def write_scalar(writer, name, v, step):
    if writer is None:
        return
    writer.add_scalar(name, v, step)

## Write val predictions curve
def write_pr_curve(writer, name, labels, predictions, step):
    if writer is None:
        return
    writer.add_pr_curve(name, labels, predictions, step)

def write_tensorboard(writer, score, result, iteration, split):
    if split == "val":
        write_scalar(writer, 'val_AUC+mAUC(weighted)', score, iteration)
        write_scalar(writer, 'val_auc', result['auc'], iteration)
        write_scalar(writer, 'val_macro_auc', result['macro_auc'], iteration)
        write_scalar(writer, 'val_max_micro_f1', result['max_micro_f1'], iteration)
        write_scalar(writer, 'val_max_macro_f1', result['max_macro_f1'], iteration)
        write_scalar(writer, 'val_p@2%', result['p@2%'], iteration)
        write_scalar(writer, 'val_mp@2%', result['mp@2%'], iteration)
        write_pr_curve(writer, 'val_pr_curve', result['pr_curve_labels'], result['pr_curve_predictions'], iteration)
    elif split == "test":
        write_scalar(writer, 'test_auc', result['auc'], iteration)
        write_scalar(writer, 'test_macro_auc', result['macro_auc'], iteration)
        write_scalar(writer, 'test_max_micro_f1', result['max_micro_f1'], iteration)
        write_scalar(writer, 'test_max_macro_f1', result['max_macro_f1'], iteration)
        write_scalar(writer, 'test_p@2%', result['p@2%'], iteration)
        write_scalar(writer, 'test_mp@2%', result['mp@2%'], iteration)
        write_pr_curve(writer, 'test_pr_curve', result['pr_curve_labels'], result['pr_curve_predictions'], iteration)

####################################################################
