#!/usr/bin/env bash
# 07/2022 sylvain.legroux@gmail.com

export OMP_NUM_THREADS=1
set -x
set -a

# DATASET
data_root=$DATA/en
dataset=${data_root}/obama
manifest_name="obama_5min"
wandb_project_name=${manifest_name}
exp_dir=${manifest_name}
manifest_filepath=${dataset}/${manifest_name}.json
output_dir=${dataset}
# TRAIN/TEST FASTPITCH
train_manifest=${output_dir}/${manifest_name}.train.json
test_manifest=${output_dir}/${manifest_name}.test.json
# TRAIN/TEST HIFIGAN
train_manifest_hifigan=${output_dir}/${manifest_name}.train.hifigan.json
test_manifest_hifigan=${output_dir}/${manifest_name}.test.hifigan.json
# fine_tune_bs=64
# n_speakers=1
synth_text="This is a test"
specgen_ckpt_dir=${exp_dir}_finetune

#./en.sh $ARGS "$@"
# 100: train/test split
# 4: fastpitch fine-tune
# 6: hifigan data prep
# 7: hifigan fine-tune
# 9: test fine-tuned models
# for stage in 100 4 6 7 9; do

for stage in 4; do
    time ./en.sh ${stage}
done
