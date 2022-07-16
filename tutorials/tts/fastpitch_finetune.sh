#!/usr/bin/env bash
export HYDRA_FULL_ERROR=1
# tuto
train="./6097_manifest_train_dur_5_mins_local.json"
val="./6097_manifest_dev_ns_all_local.json"
exp="./ljspeech_to_6097_no_mixing_5_mins"
sup_data_path=./fastpitch_sup_data
# Kareem
dataset="/home/syl20/data/en/webex_speakers/en"
train=${dataset}/kareem.json
val=${dataset}/kareem_val.json
sup_data_path=${dataset}/sup_data
exp="./kareem"
set - x

# setup paths

# finetune
python fastpitch_finetune.py --config-name=fastpitch_align_v1.05.yaml \
  train_dataset=${train} \
  validation_datasets=${val} \
  sup_data_path=${sup_data_path} \
  phoneme_dict_path=tts_dataset_files/cmudict-0.7b_nv22.01 \
  heteronyms_path=tts_dataset_files/heteronyms-030921 \
  whitelist_path=tts_dataset_files/lj_speech.tsv \
  exp_manager.exp_dir=${exp} \
  +init_from_nemo_model=./tts_en_fastpitch_align.nemo \
  +trainer.max_steps=1000 ~trainer.max_epochs \
  trainer.check_val_every_n_epoch=25 \
  model.train_ds.dataloader_params.batch_size=24 model.validation_ds.dataloader_params.batch_size=24 \
  model.n_speakers=1 model.pitch_mean=121.9 model.pitch_std=23.1 \
  model.pitch_fmin=30 model.pitch_fmax=512 model.optim.lr=2e-4 \
  ~model.optim.sched model.optim.name=adam trainer.devices=8 trainer.strategy=ddp \
  +model.text_tokenizer.add_blank_at=true \
  # trainer.strategy=ddp \