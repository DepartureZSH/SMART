import pickle
import os
import time
import os.path as op
import torch
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import smart.modeling.modeling_smart as modeling_gre

from smart.utils.logger import setup_logger
from smart.utils.misc import (mkdir, set_seed)
from smart.utils.comm import all_gather, gather_on_master, reduce_dict, get_world_size, get_rank, is_main_process
from smart.utils.save_model import save_model
from smart.utils.initizer import *
from smart.utils.optimizer import *

## Calculate the eval results
def eval_clever(args, model, data_loader, tokenizer, desc='Eval'):
    world_all_scores = {}
    model.eval()
    pred_results = {}

    loader_time = 0
    forward_time = 0
    recall_time = 0

    loader_start_time = time.time()
    rank = get_rank()
    start_time = time.time()
    all_results = []
    for step, (label_list, batch) in tqdm(enumerate(data_loader), desc=desc, disable=(rank not in [0, -1])):
        (bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
         bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list,
         bag_image_ids_list, preload_ids_list) = label_list
        batch = tuple([x.to(args.device) for x in t] for t in batch)
        img_feat, input_ids, input_mask, segment_ids = batch
        loader_end_time = time.time()
        loader_time += loader_end_time - loader_start_time

        # eval
        forward_start_time = time.time()
        with torch.no_grad():
            with torch.amp.autocast():
                logits = model(bag_input_ids=input_ids, bag_token_type_ids=segment_ids, bag_attention_mask=input_mask, bag_img_feats=img_feat,
                               bag_object_box_lists=bag_object_boxes_list,
                               bag_object_name_positions_lists=bag_object_name_positions_list,
                               bag_head_obj_idxs_list=bag_head_obj_idxs_list,
                               bag_tail_obj_idxs_list=bag_tail_obj_idxs_list,
                               bag_labels=bag_labels, attention_label_list=attention_label_list,
                               bag_image_ids_list=bag_image_ids_list, bag_key_list=bag_key_list,
                               preload_ids_list=preload_ids_list)

        forward_end_time = time.time()
        forward_time += forward_end_time - forward_start_time

        recall_start_time = time.time()

        all_logits = torch.cat(all_gather(logits.cpu()), dim=0)
        all_pair = sum(all_gather(bag_key_list), [])

        for bag_key, logits in zip(all_pair, all_logits):
            logits = logits.cpu().numpy()
            assert len(logits) == 101, f'{len(logits)}'  # NOTE: change it if necessary
            for rel_id in range(len(logits)):
                if args.num_classes == 101:
                    all_results.append({
                        'class_pair': bag_key.split('#'),
                        'relation': rel_id,
                        'score': logits[rel_id]
                    })
                elif args.num_classes == 100:
                    if rel_id == 0:
                        continue
                    all_results.append({
                        'class_pair': bag_key.split('#'),
                        'relation': rel_id,
                        'score': logits[rel_id]
                    })
        recall_end_time = time.time()
        recall_time += recall_end_time - recall_start_time

        loader_start_time = time.time()
        t = time.time() - start_time
    result = data_loader.dataset.eval(all_results)
    return result

## Calculate the eval results
def eval_smart(args, model, data_loader, tokenizer, desc='Eval'):
    world_all_scores = {}
    model.eval()
    pred_results = {}

    loader_time = 0
    forward_time = 0
    recall_time = 0

    loader_start_time = time.time()
    rank = get_rank()
    start_time = time.time()
    all_results = []
    for step, (label_list, batch) in tqdm(enumerate(data_loader), desc=desc, disable=(rank not in [0, -1])):
        (bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
         bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list,
         bag_image_ids_list, preload_ids_list) = label_list
        batch = tuple([x.to(args.device) for x in t] for t in batch)
        img_feat, caption_feat, input_ids, input_mask, segment_ids = batch
        loader_end_time = time.time()
        loader_time += loader_end_time - loader_start_time

        # eval
        forward_start_time = time.time()
        with torch.no_grad():
            with torch.amp.autocast():
                logits = model(bag_input_ids=input_ids, bag_token_type_ids=segment_ids, bag_attention_mask=input_mask, bag_img_feats=img_feat,
                               bag_caption_feats = caption_feat,
                               bag_object_box_lists=bag_object_boxes_list,
                               bag_object_name_positions_lists=bag_object_name_positions_list,
                               bag_head_obj_idxs_list=bag_head_obj_idxs_list,
                               bag_tail_obj_idxs_list=bag_tail_obj_idxs_list,
                               bag_labels=bag_labels, attention_label_list=attention_label_list,
                               bag_image_ids_list=bag_image_ids_list, bag_key_list=bag_key_list,
                               preload_ids_list=preload_ids_list)

        forward_end_time = time.time()
        forward_time += forward_end_time - forward_start_time

        recall_start_time = time.time()

        all_logits = torch.cat(all_gather(logits.cpu()), dim=0)
        all_pair = sum(all_gather(bag_key_list), [])

        for bag_key, logits in zip(all_pair, all_logits):
            logits = logits.cpu().numpy()
            assert len(logits) == 101, f'{len(logits)}'  # NOTE: change it if necessary
            for rel_id in range(len(logits)):
                if args.num_classes == 101:
                    all_results.append({
                        'class_pair': bag_key.split('#'),
                        'relation': rel_id,
                        'score': logits[rel_id]
                    })
                elif args.num_classes == 100:
                    if rel_id == 0:
                        continue
                    all_results.append({
                        'class_pair': bag_key.split('#'),
                        'relation': rel_id,
                        'score': logits[rel_id]
                    })

        recall_end_time = time.time()
        recall_time += recall_end_time - recall_start_time

        loader_start_time = time.time()
        t = time.time() - start_time
    result = data_loader.dataset.eval(all_results)
    return result

# Training process
def train_clever(args, train_loader, val_loader, test_loader, model, scheduler, optimizer, tokenizer, config, writer=None):
    rank = get_rank()
    scaler = torch.amp.GradScaler()

    best_score = 0
    global_step = 0
    iteration = 0
    log_loss = 0

    pre_eval = True
    if args.head == 'max':
        val_loader = test_loader
    if pre_eval:
        val_rtn = eval_clever(args, model, val_loader, tokenizer, desc=f'Eval Epoch-{0}/{config.num_train_steps}')

        best_score = val_rtn['auc'] + args.mAUC_weight * val_rtn['macro_auc']
        write_tensorboard(writer, epoch_score, val_rtn, iteration, "val")

        logger.info(f'Step-0 auc: {val_rtn["auc"]:.4f}, m_auc: {val_rtn["macro_auc"]:.4f}, '
                    f'micro-f1:{val_rtn["max_micro_f1"]:.4f}, '
                    f'macro-f1:{val_rtn["max_macro_f1"]:.4f}, p@2%:{val_rtn["p@2%"]:.4f}, '
                    f'mp@2%:{val_rtn["mp@2%"]:.4f}')
    if args.head == 'max':
        print(f'End eval for VRD-{args.head} evaluation')
        exit()

    for epoch in range(args.num_train_epochs):
        desc = f'Training Epoch-{epoch + 1}/{args.num_train_epochs}'
        for step, (label_list, batch) in tqdm(enumerate(train_loader), desc=desc, disable=(rank not in [0, -1])):
            global_step += 1
            model.train()
            (bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
             bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list,
             bag_image_ids_list, preload_ids_list) = label_list
            batch = tuple([x.to(args.device) for x in t] for t in batch)
            img_feat, caption_feat, input_ids, input_mask, segment_ids = batch
            # bert forward
            with torch.amp.autocast():
                loss = model(bag_input_ids=input_ids, bag_token_type_ids=segment_ids, bag_attention_mask=input_mask, bag_img_feats=img_feat,
                             bag_caption_feats=caption_feat,
                             bag_object_box_lists=bag_object_boxes_list,
                             bag_object_name_positions_lists=bag_object_name_positions_list,
                             bag_head_obj_idxs_list=bag_head_obj_idxs_list,
                             bag_tail_obj_idxs_list=bag_tail_obj_idxs_list,
                             bag_labels=bag_labels, attention_label_list=attention_label_list,
                             bag_image_ids_list=bag_image_ids_list, bag_key_list=bag_key_list,
                             preload_ids_list=preload_ids_list)

            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            # scaler.scale(loss).backward()
            log_loss += (loss / args.gradient_accumulation_steps).item()

            val_result = None

            if global_step % args.gradient_accumulation_steps == 0:
                iteration += 1

                write_scalar(writer, 'train_loss', log_loss, iteration)
                write_scalar(writer, 'learning rate', optimizer.param_groups[-1]["lr"], iteration)
                log_loss = 0

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if iteration and (iteration % 200 == 0):
                    result = eval_clever(args, model, val_loader, tokenizer, desc=f'Eval step {iteration}')

                    epoch_score = result['auc'] + args.mAUC_weight * result['macro_auc']
                    val_result = epoch_score
                    write_tensorboard(writer, epoch_score, val_rtn, iteration, "val")

                    logger.info(f'Step-{iteration} auc: {result["auc"]:.4f}, m_auc: {result["macro_auc"]:.4f}, '
                                f'micro-f1:{result["max_micro_f1"]:.4f}, '
                                f'macro-f1:{result["max_macro_f1"]:.4f}, p@2%:{result["p@2%"]:.4f}, '
                                f'mp@2%:{result["mp@2%"]:.4f}')
                    if epoch_score >= best_score:
                        best_score = epoch_score
                        save_model(args, model, tokenizer, logger, save_mode="best")

                        # Run Test Split
                        result = eval_clever(args, model, test_loader, tokenizer, desc=f'Test step-{iteration}')
                        write_tensorboard(writer, epoch_score, val_rtn, iteration, "test")
                        pickle.dump(result['results'], open(f'{args.output_dir}/best_results.pkl', 'wb'))

            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= 3:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                return model
    return model

# Training process
def train_smart(args, train_loader, val_loader, test_loader, model, scheduler, optimizer, tokenizer, config, writer=None):
    rank = get_rank()
    scaler = torch.amp.GradScaler()

    best_score = 0
    global_step = 0
    iteration = 0
    log_loss = 0

    pre_eval = True
    if args.head == 'max':
        val_loader = test_loader
    if pre_eval:
        val_rtn = eval_smart(args, model, val_loader, tokenizer, desc=f'Eval Epoch-{0}/{config.num_train_steps}')

        best_score = val_rtn['auc'] + args.mAUC_weight * val_rtn['macro_auc']
        write_tensorboard(writer, epoch_score, val_rtn, iteration, "val")

        logger.info(f'Step-0 auc: {val_rtn["auc"]:.4f}, m_auc: {val_rtn["macro_auc"]:.4f}, '
                    f'micro-f1:{val_rtn["max_micro_f1"]:.4f}, '
                    f'macro-f1:{val_rtn["max_macro_f1"]:.4f}, p@2%:{val_rtn["p@2%"]:.4f}, '
                    f'mp@2%:{val_rtn["mp@2%"]:.4f}')

    for epoch in range(args.num_train_epochs):
        desc = f'Training Epoch-{epoch + 1}/{args.num_train_epochs}'
        for step, (label_list, batch) in tqdm(enumerate(train_loader), desc=desc, disable=(rank not in [0, -1])):
            global_step += 1
            model.train()
            (bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
             bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list,
             bag_image_ids_list, preload_ids_list) = label_list
            batch = tuple([x.to(args.device) for x in t] for t in batch)
            img_feat, caption_feat, input_ids, input_mask, segment_ids = batch
            # bert forward
            with torch.amp.autocast():
                loss = model(bag_input_ids=input_ids, bag_token_type_ids=segment_ids, bag_attention_mask=input_mask, bag_img_feats=img_feat,
                             bag_caption_feats=caption_feat,
                             bag_object_box_lists=bag_object_boxes_list,
                             bag_object_name_positions_lists=bag_object_name_positions_list,
                             bag_head_obj_idxs_list=bag_head_obj_idxs_list,
                             bag_tail_obj_idxs_list=bag_tail_obj_idxs_list,
                             bag_labels=bag_labels, attention_label_list=attention_label_list,
                             bag_image_ids_list=bag_image_ids_list, bag_key_list=bag_key_list,
                             preload_ids_list=preload_ids_list)

            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            # scaler.scale(loss).backward()
            log_loss += (loss / args.gradient_accumulation_steps).item()

            val_result = None

            if global_step % args.gradient_accumulation_steps == 0:
                iteration += 1

                write_scalar(writer, 'train_loss', log_loss, iteration)
                write_scalar(writer, 'learning rate', optimizer.param_groups[-1]["lr"], iteration)
                log_loss = 0

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if iteration and (iteration % 200 == 0):
                    result = eval_smart(args, model, val_loader, tokenizer, desc=f'Eval step {iteration}')

                    epoch_score = result['auc'] + args.mAUC_weight * result['macro_auc']
                    val_result = epoch_score
                    write_tensorboard(writer, epoch_score, val_rtn, iteration, "val")

                    logger.info(f'Step-{iteration} auc: {result["auc"]:.4f}, m_auc: {result["macro_auc"]:.4f}, '
                                f'micro-f1:{result["max_micro_f1"]:.4f}, '
                                f'macro-f1:{result["max_macro_f1"]:.4f}, p@2%:{result["p@2%"]:.4f}, '
                                f'mp@2%:{result["mp@2%"]:.4f}')
                    if epoch_score >= best_score:
                        best_score = epoch_score
                        save_model(args, model, tokenizer, logger, save_mode="best")

                        # Run Test Split
                        result = eval_smart(args, model, test_loader, tokenizer, desc=f'Test step-{iteration}')
                        write_tensorboard(writer, epoch_score, val_rtn, iteration, "test")
                        pickle.dump(result['results'], open(f'{args.output_dir}/best_results.pkl', 'wb'))

            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= 3:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                return model
    return model

def Model(args, local_rank):
    if args.model == "CLEVER":
        from smart.modeling.pytorch_transformers import BertTokenizer, BertConfig
        from smart.modeling.modeling_clever import BagModel
        ######################################################################################################
        # Load pretrained model and tokenizer
        config_class, model_class, tokenizer_class = BertConfig, BagModel, BertTokenizer
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        # config.hidden_dropout_prob = 0.6
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        config.loss_type = "cls"

        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)

        print(f'Load from {checkpoint}')

        ######################################################################################################
        # Model weight loading
        if modeling_gre.head == 'att' and args.pretrained_weight:
            model.load_state_dict(torch.load(args.pretrained_weight, map_location='cpu'), strict=False)
        elif modeling_gre.head in ['avg', 'max'] and args.VRD_weight:
            model.load_state_dict(torch.load(args.VRD_weight, map_location='cpu'), strict=False)

        ######################################################################################################
        # distributed
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        ######################################################################################################
        logger.info(args)
        return model, tokenizer, config

    elif args.model == "SMART":
        from smart.modeling.pytorch_transformers import BertTokenizer, BertConfig
        from smart.modeling.modeling_smart import BagModel
        ######################################################################################################
        # Load pretrained model and tokenizer
        config_class, model_class, tokenizer_class = BertConfig, BagModel, BertTokenizer
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        # config.hidden_dropout_prob = 0.6
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        config.loss_type = "cls"

        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)

        print(f'Load from {checkpoint}')

        ######################################################################################################
        # Model weight loading
        # if modeling_gre.head == 'att' and args.pretrained_weight:
        #     model.load_state_dict(torch.load(args.pretrained_weight, map_location='cpu'), strict=False)
        # elif modeling_gre.head in ['avg', 'max'] and args.VRD_weight:
        #     model.load_state_dict(torch.load(args.VRD_weight, map_location='cpu'), strict=False)

        ######################################################################################################
        # distributed
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        ######################################################################################################
        logger.info(args)
        return model, tokenizer, config

def main():
    args = get_parser()

    modeling_gre.sfmx_t = args.sfmx_t
    modeling_gre.attention_w = args.attention_w
    modeling_gre.head = args.head
    modeling_gre.select_size = args.select_size
    modeling_gre.loss_w_t = args.loss_w_t

    global logger

    ######################################################################################################
    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=int(os.environ["LOCAL_RANK"]))
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args, logger)

    ######################################################################################################
    # Tensorboard writer setting
    writer = None
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(
            log_dir=os.path.join(output_dir, 'summary'),
            flush_secs=10
        )

    ######################################################################################################
    # Model loading
    model, tokenizer, config = Model(args, local_rank)

    ######################################################################################################
    # data loading
    if local_rank != 0:
        torch.distributed.barrier()

    train_loader = make_data_loader(
        args, logger, f'{args.train_dir}/train_bag_data.json', 'train', tokenizer,
        is_distributed=args.distributed,
        is_train=True)
    test_loader = make_data_loader(
        args, logger, f'{args.train_dir}/test_bag_data.json', 'test', tokenizer,
        is_distributed=args.distributed,
        is_train=False)
    val_loader = make_data_loader(
        args, logger, f'{args.train_dir}/val_bag_data.json', 'val', tokenizer, is_distributed=args.distributed,
        is_train=False)

    if local_rank == 0:
        torch.distributed.barrier()

    print(f'Train-size: {len(train_loader.dataset)} bags, {len(train_loader)} batches, '
          f'Test-size: {len(test_loader.dataset)} bags, {len(test_loader)} batches')
    
    ######################################################################################################

    config.learning_rate = args.learning_rate
    config.weight_decay = 0.01
    config.betas = [0.9, 0.98]
    config.num_train_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    config.lr_mul = 1.0
    config.warmup_steps = int(0.1 * config.num_train_steps)

    ######################################################################################################
    # Optimizer setting
    optimizer = build_optimizer(model, config)
    print(f'Warm up step {args.warmup_steps}')
    scheduler = WarmupReduceLROnPlateau(
        optimizer,
        0.1,
        warmup_factor=0.1,
        warmup_iters=args.warmup_steps,
        warmup_method=args.scheduler,
        T_max=3000,
        eta_min=0,
        patience=3,
        threshold=0.001,
        cooldown=0,
        logger=logger,
    )
    ######################################################################################################
    if args.model == "CLEVER":
        model = train_clever(args, train_loader, val_loader, test_loader, model, scheduler, optimizer, tokenizer, config,
                  writer=writer)
    elif args.model == "SMART":
        model = train_smart(args, train_loader, val_loader, test_loader, model, scheduler, optimizer, tokenizer, config,
                    writer=writer)
    save_model(args, model, tokenizer, logger, save_mode="final")
    ######################################################################################################

if __name__ == "__main__":
    main()