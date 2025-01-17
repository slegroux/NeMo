#!/usr/bin/env bash
# 07/2022 sylvain.legroux@gmail.com

export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PS1='\u@\h: '
set -x

stage=$1

: ${DATA:="/home/syl20/data"}
: ${sup_data_folder:=${dataset}/sup_data}
: ${output_dir:=${dataset}}
: ${config_path:="conf"}
: ${fastpitch_config:="fastpitch_align_v1.05.yaml"}
: ${hifigan_config:="hifigan.yaml"}
: ${wandb_project_name:=$(basename ${dataset})}
: ${wandb_run_name:=${fastpitch_config}}
: ${exp_dir:=${wandb_project_name}}
: ${sid:="dc52e6a0-a9b8-3e0a-ace7-78b00108a7b7"} # kareem
: ${total_dur:=300}
: ${train_size:=0.98}
: ${min_dur:=1.0}
: ${max_dur:=10.0}

# additional files for phonemes, heteronyms & abbrv
: ${helper_dir:="/home/syl20/slgNM/tutorials/tts/tts_dataset_files"}
: ${pretrained_model:="./tts_en_fastpitch_align.nemo"}

if [ ${stage} -eq 0 ]; then
    # manifest + normalization
    python ${dataset}/metadata2manifest.py --sid ${sid} ${metadata_filepath} ${pre_manifest_filepath}
fi

if [ ${stage} -eq 100 ]; then
    echo "[INFO] filter manifest duration"
    python ~/maui/scripts/datasets/manifest_filter.py --total_dur ${total_dur} --min_dur ${min_dur} --max_dur ${max_dur} ${pre_manifest_filepath} ${manifest_filepath}
    python ~/maui/scripts/datasets/manifest_duration.py ${manifest_filepath}
    echo "[INFO] train/test split"
    python /home/syl20/maui/scripts/datasets/train_test_split_file.py --train_size ${train_size} --output_dir ${output_dir} ${manifest_filepath}
fi

if [ ${stage} -eq 1 ]; then
# only for DE
    python phonemize.py --train ${train} --test ${test} --val ${val} --lang 'de'
fi    

if [ ${stage} -eq 2 ]; then
    # extract from original full dataset manifest
    # extract pitch & pitch stats to stabilize training
    # not necessary if only fine-tuning
    # 1067 Pitch mean:117.12761688232422, Std: 23.40104103088379, Min:65.4063949584961, Max:1028.5239257812
    python extract_sup_data.py \
        --config-path ${config_path} \
        --config-name "ds_for_fastpitch_align.yaml" \
        manifest_filepath=${manifest_filepath} \
        sup_data_path=${sup_data_folder}
fi

pitch_mean=117.12761688232422
pitch_std=23.40104103088379
if [ ${stage} -eq 3 ]; then
    # train
    python fastpitch.py --config-path ${config_path} --config-name ${fastpitch_config} \
        model.train_ds.dataloader_params.batch_size=32 \
        model.validation_ds.dataloader_params.batch_size=32 \
        train_dataset=${train_manifest} \
        validation_datasets=${test_manifest} \
        sup_data_path=${sup_data_folder} \
        whitelist_path=${helper_dir}/lj_speech.tsv \
        exp_manager.exp_dir=${exp_dir} \
        trainer.max_epochs=1000 \
        +exp_manager.create_wandb_logger=true \
        +exp_manager.wandb_logger_kwargs.name=${fastpitch_config} \
        +exp_manager.wandb_logger_kwargs.project=${wandb_project_name}
        # pitch_mean=${pitch_mean} \
        # pitch_std=${pitch_std} \
fi

if [ ${stage} -eq 4 ]; then
    # note: computes sup_data if not already computed
    # just use librosa defaults for now
    # pitch_mean=117.12761688232422
    # pitch_std=23.40104103088379
    #  model.n_speakers=1 model.pitch_mean=121.9 model.pitch_std=23.1 \
    #  model.pitch_fmin=30 model.pitch_fmax=512 model.optim.lr=2e-4 \
    # pitch_fmin=30
    # pitch_fmax=512
    : ${n_speakers:=1}
    # : {fine_tune_bs:=48} #24
    : ${fastpitch_fine_tune_bs:=48}
    : ${fastpich_finetune_steps:=1000}

    python fastpitch_finetune.py --config-name=${fastpitch_config} \
        train_dataset=${train_manifest} \
        validation_datasets=${test_manifest} \
        sup_data_path=${sup_data_folder} \
        phoneme_dict_path=${helper_dir}/cmudict-0.7b_nv22.07 \
        heteronyms_path=${helper_dir}/heteronyms-030921 \
        whitelist_path=${helper_dir}/lj_speech.tsv \
        exp_manager.exp_dir=${exp_dir}_finetune_fastpitch \
        +init_from_nemo_model=${pretrained_model} \
        +trainer.max_steps=${fastpich_finetune_steps} ~trainer.max_epochs \
        trainer.check_val_every_n_epoch=25 \
        model.train_ds.dataloader_params.batch_size=${fastpitch_fine_tune_bs} \
        model.validation_ds.dataloader_params.batch_size=${fastpitch_fine_tune_bs} \
        model.n_speakers=${n_speakers} \
        model.optim.lr=2e-4 \
        ~model.optim.sched model.optim.name=adam trainer.devices=8 trainer.strategy=ddp \
        +model.text_tokenizer.add_blank_at=true \
        +exp_manager.create_wandb_logger=true \
        +exp_manager.wandb_logger_kwargs.name=${fastpitch_config} \
        +exp_manager.wandb_logger_kwargs.project=${wandb_project_name}_finetune
        # model.pitch_mean=${pitch_mean} model.pitch_std=${pitch_std} \
        # model.pitch_fmin=${pitch_fmin} model.pitch_fmax=${pitch_fmax} \
fi

if [ ${stage} -eq 5 ]; then
    # ljspeech_to_6097_no_mixing_5_mins/FastPitch/2022-07-06_19-07-07/checkpoints/FastPitch--v_loss=1.9898-epoch=49-last.ckpt
    # specgen_ckpt_dir=${exp_dir}_finetune
    specgen_ckpt_dir=kareem
    python fastpitch_synth.py \
        --specgen_ckpt_dir ${specgen_ckpt_dir} \
        --vocoder "tts_hifigan" \
        --input_text "This is a test, and it's not too good." \
        --output_audio "this_is_a_test.kareem.wav" \
        --sr 22050
fi

if [ ${stage} -eq 6 ]; then
    # hifigan data prep for train & validation sets
    mels_dir=${sup_data_folder}/mels
    # specgen_ckpt_dir=${exp_dir}_finetune
    # specgen_ckpt_dir=ljspeech_to_6097_no_mixing_5_mins
    : ${specgen_ckpt_dir:=${exp_dir}_finetune_fastpitch}

    # echo $(basename ${manifest%.*}).hifigan.json
    python hifigan_dataprep.py \
        --manifest_path ${train_manifest} \
        --mels_dir ${mels_dir}/train \
        --specgen_ckpt_dir ${specgen_ckpt_dir} \
        --hifigan_manifest_path ${dataset}/$(basename ${train_manifest%.*}).hifigan.json
    
    python hifigan_dataprep.py \
        --manifest_path ${test_manifest} \
        --mels_dir ${mels_dir}/test \
        --specgen_ckpt_dir ${specgen_ckpt_dir} \
        --hifigan_manifest_path ${dataset}/$(basename ${test_manifest%.*}).hifigan.json

fi

if [ ${stage} -eq 7 ]; then
    # hifigan fine-tuning
    # train=${dataset}/kareem.hifigan.json
    # val=${dataset}/kareem_val.hifigan.json
    # train="./hifigan_train_ft.json"
    # val="./hifigan_val_ft.json"
    : ${pretrained_checkpoint:="tts_hifigan"} # pre-trained nvidia model
    # exp_dir="test_finetune_hifigan"
    : ${hifigan_finetune_bs:=64}
    : ${hifigan_finetune_steps:=1000}

    python hifigan_finetune.py \
        --config-name=${hifigan_config} \
        model.train_ds.dataloader_params.batch_size=${hifigan_finetune_bs} \
        model.max_steps=${hifigan_finetune_steps} \
        model.optim.lr=0.00001 \
        ~model.optim.sched \
        train_dataset=${train_manifest_hifigan} \
        validation_datasets=${test_manifest_hifigan} \
        exp_manager.exp_dir=${exp_dir}_finetune_hifigan \
        +init_from_pretrained_model=${pretrained_checkpoint} \
        trainer.check_val_every_n_epoch=10 \
        trainer.devices=8 \
        model/train_ds=train_ds_finetune \
        model/validation_ds=val_ds_finetune \
        ++exp_manager.create_wandb_logger=true \
        +exp_manager.wandb_logger_kwargs.name=${hifigan_config} \
        +exp_manager.wandb_logger_kwargs.project=${wandb_project_name}_finetune_hifigan
fi

if [ ${stage} -eq 8 ]; then
    python get_best_checkpoint.py \
        --base_dir . \
        --checkpoint_dir ${exp_dir}_finetune_hifigan \
        --model_name "HifiGan"
fi

if [ ${stage} -eq 9 ]; then
    : ${specgen_ckpt_dir:=${exp_dir}_finetune_fastpitch}
    : ${vocoder_ckpt_dir:=${exp_dir}_finetune_hifigan}
    : ${speed_factor:=1.0}
    : ${pitch_factor:=1.0}
    : ${pitch_scale:=0.0}
    : ${synth_text:="This is an example of an adapted speaker"}

    text_underscore=$(echo ${synth_text}|sed 's| |_|g')

    python fastpitch_synth.py \
        --specgen_ckpt_dir ${specgen_ckpt_dir} \
        --vocoder_ckpt_dir ${vocoder_ckpt_dir} \
        --speed_factor ${speed_factor} \
        --pitch_factor ${pitch_factor} \
        --pitch_scale ${pitch_scale} \
        --input_text "${synth_text}" \
        --output_audio "${exp_dir}_${text_underscore}-sp${speed_factor}-pf${pitch_factor}-ps${pitch_scale}.wav" \
        --sr 22050
fi
