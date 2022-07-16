#!/usr/bin/env bash
set -x
export HYDRA_FULL_ERROR=1
train="./hifigan_train_ft.json"
val="./hifigan_val_ft.json"
exp="hifigan_ft"
# train="/home/syl20/data/en/6097_5_mins/manifest.train.hifigan.json"
# val="/home/syl20/data/en/6097_5_mins/manifest.test.hifigan.json"
# train="/home/syl20/data/en/6097_5_mins/6097_manifest_train_dur_5_mins_local.json"
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
    model/train_ds=train_ds_finetune \
    model/validation_ds=val_ds_finetune

# trainer.devices=8 \