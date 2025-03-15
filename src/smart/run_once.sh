SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DATA_DIR=$(realpath -m "${SCRIPT_DIR}/../../data")
MODEL_DIR=$(realpath -m "${SCRIPT_DIR}/../../pretrained_models")
OUTPUT_DIR=$(realpath -m "${SCRIPT_DIR}/../../output")
SAMPLE_PER_CARD=2 # 1,2,4,8
ACCUMULATE_STEP=32 # 8,16,32,64
WARMUP_STEP=1000
NUM_CLASS=101 # Exclude blank label: 100; Not exclude blan label: 101
MODEL=CLEVER # [CLEVER, SMART]
HEAD=att # CLEVER: [att, origin_att, avg, one], SMART: [simi_att, Custom]
SCHEDULER=linear # [constant, linear, cosine]
seed=42
lr=7e-5
mAUC_weight=3

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 10226 run_bag.py \
  --model ${MODEL}\
  --num_classes ${NUM_CLASS} \
  --eval_model_dir ${MODEL_DIR}/pretrained_base/ \
  --pretrained_weight ${MODEL_DIR}/pretrained_avg_model/model.bin \
  --do_train \
  --train_dir ${DATA_DIR} \
  --test_dir ${DATA_DIR} \
  --do_lower_case \
  --learning_rate ${lr} \
  --warmup_steps ${WARMUP_STEP} \
  --scheduler ${SCHEDULER} \
  --gradient_accumulation_steps ${ACCUMULATE_STEP} \
  --per_gpu_train_batch_size ${SAMPLE_PER_CARD} \
  --per_gpu_eval_batch_size ${SAMPLE_PER_CARD} \
  --num_train_epochs 100 \
  --freeze_embedding \
  --output_dir ${OUTPUT_DIR}/${MODEL}/${HEAD}_lr${lr}_bsz-10card-${SAMPLE_PER_CARD}-${ACCUMULATE_STEP}_seed${seed}_warm${WARMUP_STEP}_mAUC${mAUC_weight} \
  --sfmx_t 11 \
  --attention_w 0.0 \
  --head ${HEAD}\
  --real_bag_size 50 \
  --select_size 50 \
  --seed ${seed} \
  --mAUC_weight ${mAUC_weight}
