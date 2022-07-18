#!/usr/bin/env bash
# 07/2022 sylvain.legroux@gmail.com

export OMP_NUM_THREADS=1
# set -x
set -a

# DATASET
data_root=$DATA/en/
dataset=${data_root}/webex_speakers/en
pre_manifest_filepath=${dataset}/mayada.json
manifest_name="mayada_5min"
wandb_project_name=${manifest_name}
exp_dir=${manifest_name}
manifest_filepath=${dataset}/${manifest_name}.json
output_dir=${dataset}
sup_data_folder=${dataset}/${manifest_name}/sup_data
# TRAIN/TEST FASTPITCH
train_size=0.97
train_manifest=${output_dir}/${manifest_name}.train.json
test_manifest=${output_dir}/${manifest_name}.test.json
# TRAIN/TEST HIFIGAN
train_manifest_hifigan=${output_dir}/${manifest_name}.train.hifigan.json
test_manifest_hifigan=${output_dir}/${manifest_name}.test.hifigan.json
# n_speakers=1
synth_text="Good morning. This is a new method to generate voices from very little data."
# specgen_ckpt_dir=${exp_dir}_finetune
# BATCH SIZES
fastpitch_fine_tune_bs=48
hifigan_finetune_bs=48
# FILTER UTTERANCES BY DURATION
min_dur=3
max_dur=10

#./en.sh $ARGS "$@"
# 100: train/test split
# 4: fastpitch fine-tune
# 6: hifigan data prep
# 7: hifigan fine-tune
# 9: test fine-tuned models
# for stage in 100 4 6 7 9; do

for stage in 100 4 6 7 9; do
    time ./en.sh ${stage} | tee mayada.log
done

# TODO:
# try changing pitch & std from data prep for re-synthesis
# example that adapts to context with pitch & al
# experiments with mixing speakers in one model