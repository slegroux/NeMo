#!/usr/bin/env bash
# 07/2022 sylvain.legroux@gmail.com

export OMP_NUM_THREADS=1
set -x
set -a

# DATASET
data_root=$DATA/en
dataset=${data_root}/6097_5_mins
wandb_project_name=$(basename ${dataset})
exp_dir=${wandb_project_name}
manifest_filepath=${dataset}/manifest.json
output_dir=${dataset}
# TRAIN/TEST FASTPITCH
train_manifest=${output_dir}/manifest.train.json
test_manifest=${output_dir}/manifest.test.json
# TRAIN/TEST HIFIGAN
train_manifest_hifigan=${output_dir}/manifest.train.hifigan.json
test_manifest_hifigan=${output_dir}/manifest.test.hifigan.json
# fine_tune_bs=64
fastpitch_fine_tune_bs=64
hifigan_finetune_bs=64
# n_speakers=1

#./en.sh $ARGS "$@"
# train/test split
# time ./en.sh 100
# echo "[INFO] fastpitch fine-tune"
time ./en.sh 4
# echo "[INFO] hifigan data prep"
# time ./en.sh 6
# echo "[INFO] hifigan fine-tune"
# time ./en.sh 7
# echo "[INFO] test fine-tuned models"
# synth_text="She put a scarf"
# specgen_ckpt_dir=${exp_dir}_finetune
# time ./en.sh 9

# 100: train/test split
# 4: fastpitch fine-tune
# 6: hifigan data prep
# 7: hifigan fine-tune
# 9: test fine-tuned models
# for stage in 100 4 6 7 9; do

# for stage in 100 4 6 7 9; do
#     time ./en.sh ${stage} | tee kareem.log
# done