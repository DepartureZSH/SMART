SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DATA_DIR=$(realpath -m "${SCRIPT_DIR}/../../data")
##################################################
# DATA_DIR=$(realpath -m "${SCRIPT_DIR}/../../../CLEVER/MiniCLEVER/CLEVER/data")
##################################################
MODEL_DIR=$(realpath -m "${SCRIPT_DIR}/../../pretrained_models")
OUTPUT_DIR=$(realpath -m "${SCRIPT_DIR}/../../output")
SAMPLE_PER_CARD=1
ACCUMULATE_STEP=8
WARMUP_STEP=600
NUM_CLASS=101 # Exclude blank label: 100; Not exclude blan label: 101
MODEL=CLEVER # [CLEVER, SMART]
HEAD=att # CLEVER: [att, origin_att, avg, one], SMART: [simi_att, Custom]
SCHEDULER=cosine # [constant, linear, cosine]
seed=42

for lr in {4e-5,5e-5,6e-5}; do
  for mAUC_weight in {1,3,5}; do
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 10225 run_bag.py \
      --model ${MODEL}\
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
      --output_dir ${OUTPUT_DIR}/${HEAD}_lr${lr}_bsz-10card-${SAMPLE_PER_CARD}-${ACCUMULATE_STEP}_seed${seed}_warm${WARMUP_STEP}_mAUC${mAUC_weight} \
      --sfmx_t 11 \
      --attention_w 0.0 \
      --head ${HEAD}\
      --real_bag_size 50 \
      --select_size 50 \
      --seed ${seed} \
      --mAUC_weight ${mAUC_weight}
  done
done
