#!/usr/bin/env bash
# 07/2022 sylvain.legroux@gmail.com

export OMP_NUM_THREADS=1
# set -x
set -a

# DATA PATHS
# users=("rishav" "ljspeech")

new_speaker='rishav'
old_speaker='ljspeech'
speakers="${new_speaker}_${old_speaker}"
data_root=$DATA/en/
dataset=${data_root}/webex_speakers/en

# FILTER UTTERANCES BY DURATION
min_dur=1.0
max_dur=10.0
total_dur=300 #5min

# pre_manifest_filepath=${dataset}/${user}.json
manifest_name="${speakers}_${total_dur}"
wandb_project_name=${manifest_name}
exp_dir=${manifest_name}
manifest_filepath=${dataset}/${manifest_name}.json
output_dir=${dataset}
sup_data_folder=${dataset}/${manifest_name}/sup_data

# TRAIN/TEST FASTPITCH
train_size=0.99
train_manifest=${output_dir}/${manifest_name}.train.json
test_manifest=${output_dir}/${manifest_name}.test.json

# TRAIN/TEST HIFIGAN ON NEW SPEAKER
manifest_name_hifigan="rishav_${total_dur}"
train_manifest_hifigan=${output_dir}/${manifest_name_hifigan}.train.hifigan.json
test_manifest_hifigan=${output_dir}/${manifest_name_hifigan}.test.hifigan.json
synth_text="how long they were on the meeting for"
# specgen_ckpt_dir=${exp_dir}_finetune
# n_speakers=1

# TRAINING
fastpitch_config="fastpitch_align_multi_v1.05.yaml"
fastpitch_fine_tune_bs=64
fastpich_finetune_steps=1000
hifigan_config="hifigan.yaml"
hifigan_finetune_bs=64
hifigan_finetune_steps=1000
n_speakers=2

wandb_run_name="FP-batch:${fastpitch_fine_tune_bs}-steps:${fastpich_finetune_steps} \
    HFG-batch:${hifigan_finetune_bs}-steps:${hifigan_finetune_steps}-spks:${n_speakers}"

#./en.sh $ARGS "$@"
# 100: train/test split
# 4: fastpitch fine-tune
# 6: hifigan data prep
# 7: hifigan fine-tune
# 9: test fine-tuned models
echo "[INFO] concat multiple speakers"
user=${old_speaker}
sid=0
python ~/maui/scripts/datasets/manifest_filter.py --total_dur ${total_dur} --sid ${sid} --min_dur ${min_dur} --max_dur ${max_dur} ${dataset}/${user}.json ${dataset}/${user}_${total_dur}.json
python ~/maui/scripts/datasets/manifest_duration.py ${dataset}/${user}_${total_dur}.json
user=${new_speaker}
sid=1
python ~/maui/scripts/datasets/manifest_filter.py --total_dur ${total_dur} --sid ${sid} --min_dur ${min_dur} --max_dur ${max_dur} ${dataset}/${user}.json ${dataset}/${user}_${total_dur}.json
python ~/maui/scripts/datasets/manifest_duration.py ${dataset}/${user}_${total_dur}.json    

cat ${dataset}/${old_speaker}_${total_dur}.json ${dataset}/${new_speaker}_${total_dur}.json | shuf > ${manifest_filepath}

echo "[INFO] train/test split"
python /home/syl20/maui/scripts/datasets/train_test_split_file.py --train_size ${train_size} --output_dir ${output_dir} ${manifest_filepath}

for stage in 4 6 7 9; do
    time ./en.sh ${stage} > >(tee -a ${user}_stdout.log) 2> >(tee -a ${user}_stderr.log >&2)
done
