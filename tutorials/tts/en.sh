#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
set -x

stage=$1
data_root=$DATA/en
# dataset=${data_root}/webex_speakers/en
# metadata=${dataset}/kareem.txt
# manifest=${dataset}/kareem.json
# manifest=tests/data/asr/an4_val.json
dataset=${data_root}/6097_5_mins
manifest_filepath=${dataset}/manifest.json
sup_data_folder=${dataset}/sup_data
whitelist=whitelists/lj_speech.tsv
config_path=conf
fastpitch_config="fastpitch_align_22050"
wandb_project_name=$(basename ${dataset})
wandb_run_name=${fastpitch_config}
exp_dir=${wandb_project_name}

if [ ${stage} -eq 0 ]; then
    # manifest + normalization
    python ${dataset}/metadata2manifest.py --input ${metadata} --output ${manifest}
fi

if [ ${stage} -eq 1 ]; then
# only for DE
    python phonemize.py --train ${train} --test ${test} --val ${val} --lang 'de'
fi    

if [ ${stage} -eq 2 ]; then
    # extract pitch & pitch stats to stabilize training
    # not necessary if only fine-tuning
    python extract_sup_data.py \
        --config-path ${config_path} \
        --config-name "ds_for_fastpitch_align.yaml" \
        manifest_filepath=${manifest_filepath} \
        sup_data_path=${sup_data_folder}

    # python compute_sup_data.py \
    #     --sup_data_folder ${sup_data_folder}\
    #     --manifest_filepath ${manifest}
    # Pitch mean:117.12761688232422, Std: 23.40104103088379, Min:65.4063949584961, Max:1028.5239257812
    # KAREEM: PITCH_MEAN, PITCH_STD = 143.84454345703125, 24.80004119873047
fi

pitch_mean=132.524658203125
pitch_std=37.389366149902344
if [ ${stage} -eq 3 ]; then
    # train
    python fastpitch.py --config-path ${config_path} --config-name ${fastpitch_config} \
        model.train_ds.dataloader_params.batch_size=32 \
        model.validation_ds.dataloader_params.batch_size=32 \
        train_dataset=${dataset}/kareem.json \
        validation_datasets=${dataset}/kareem_val.json \
        sup_data_path=${sup_data_folder} \
        whitelist_path=${whitelist} \
        exp_manager.exp_dir=${exp_dir} \
        trainer.max_epochs=1000 \
        pitch_mean=${pitch_mean} \
        pitch_std=${pitch_std} \
        +exp_manager.create_wandb_logger=true \
        +exp_manager.wandb_logger_kwargs.name=${wandb_run_name} \
        +exp_manager.wandb_logger_kwargs.project=${wandb_project_name}
fi

if [ ${stage} -eq 4 ]; then
    finetune_conf=fastpitch_align_v1.05.yaml
    helper_dir=/home/syl20/slgNM/tutorials/tts/tts_dataset_files
    wandb_project_name='fastpitch_finetune'
    wandb_run_name="kareem"
    nemo_model=./tts_en_fastpitch_align.nemo
    python fastpitch_finetune.py --config-name=${finetune_conf} \
        train_dataset=${dataset}/kareem.json \
        validation_datasets=${dataset}/kareem_val.json \
        sup_data_path=${sup_data_folder} \
        phoneme_dict_path=${helper_dir}/cmudict-0.7b_nv22.01 \
        heteronyms_path=${helper_dir}/heteronyms-030921 \
        whitelist_path=${helper_dir}/lj_speech.tsv \
        exp_manager.exp_dir=${exp_dir} \
        +init_from_nemo_model=${nemo_model} \
        +trainer.max_steps=1000 ~trainer.max_epochs \
        trainer.check_val_every_n_epoch=25 \
        model.train_ds.dataloader_params.batch_size=24 model.validation_ds.dataloader_params.batch_size=24 \
        model.n_speakers=1 model.pitch_mean=${pitch_mean} model.pitch_std=${pitch_std} \
        model.pitch_fmin=30 model.pitch_fmax=512 model.optim.lr=2e-4 \
        ~model.optim.sched model.optim.name=adam trainer.devices=8 trainer.strategy=ddp \
        +model.text_tokenizer.add_blank_at=true \
        +exp_manager.create_wandb_logger=true \
        +exp_manager.wandb_logger_kwargs.name=${wandb_run_name} \
        +exp_manager.wandb_logger_kwargs.project=${wandb_project_name}
fi

if [ ${stage} -eq 5 ]; then
    # specgen_ckpt_dir="ljspeech_to_6097_no_mixing_5_mins"
    specgen_ckpt_dir="kareem"
    python fastpitch_synth.py \
        --specgen_ckpt_dir ${specgen_ckpt_dir} \
        --vocoder "tts_hifigan" \
        --input_text "This is a test of how well this new model is behaving." \
        --output_audio "this_is_a_test.wav" \
        --sr 22050
fi

if [ ${stage} -eq 6 ]; then
    specgen_ckpt_dir=result
    manifest_path="6097_manifest_train_dur_5_mins_local.json"
    hifigan_manifest_path="6097_manifest_train_dur_5_mins_local_hifigan.json"
    mels_dir="6097_manifest_train_dur_5_mins_local_mels"
    python hifigan_dataprep.py \
        --manifest_path ${manifest_path} \
        --mels_dir ${mels_dir} \
        --specgen_ckpt_dir ${specgen_ckpt_dir} \
        --hifigan_manifest_path ${hifigan_manifest_path}
fi

if [ ${stage} -eq 7 ]; then
    train="./hifigan_train_ft.json"
    val="./hifigan_val_ft.json"
    exp="hifigan_ft"

    python hifigan_finetune.py \
        --config-name=hifigan.yaml \
        model.train_ds.dataloader_params.batch_size=32 \
        model.max_steps=1000 \
        model.optim.lr=0.00001 \
        ~model.optim.sched \
        train_dataset=${train} \
        validation_datasets=${val} \
        exp_manager.exp_dir=${exp} \
        +init_from_pretrained_model=tts_hifigan \
        trainer.check_val_every_n_epoch=10 \
        trainer.devices=8 \
        model/train_ds=train_ds_finetune \
        model/validation_ds=val_ds_finetune
fi