#!/usr/bin/env bash
export OMP_NUM_THREADS=1

stage=$1
data_root=$DATA/de
dataset=thorsten-de
train=${data_root}/${dataset}/train_manifest
test=${data_root}/${dataset}/test_manifest
val=${data_root}/${dataset}/val_manifest
config_path=conf
fastpitch_config="fastpitch_align_22050"
wandb_project_name="fastpitch_de"
wandb_run_name="test"

set -x

if [ ${stage} -eq 0 ]; then
# manifest + normalization
    python get_data.py \
            --data-root ${data_root} \
            --val-size 0.1 \
            --test-size 0.2
fi


if [ ${stage} -eq 1 ]; then
    # 
    python phonemize.py --train ${train} --test ${test} --val ${val} --lang 'de'
fi    

if [ ${stage} -eq 2 ]; then
# extract pitch & pitch stats to stabilize training
    python extract_sup_data.py \
        --config-path ${config_path} \
        --config-name ds_for_fastpitch_align_de.yaml \
        manifest_filepath=${data_root}/${dataset}/train_manifest_phonemes.json \
        sup_data_path=${data_root}/${dataset}/phonemes/
fi


if [ ${stage} -eq 3 ]; then
# train
    python fastpitch.py --config-path ${config_path} --config-name ${fastpitch_config} \
        model.train_ds.dataloader_params.batch_size=32 \
        model.validation_ds.dataloader_params.batch_size=32 \
        train_dataset=${data_root}/${dataset}/train_manifest_phonemes.json \
        validation_datasets=${data_root}/${dataset}/val_manifest_phonemes.json \
        sup_data_path=${data_root}/${dataset}/phonemes/ \
        whitelist_path=whitelists/de.tsv \
        exp_manager.exp_dir=result \
        trainer.max_epochs=1 \
        pitch_mean=132.524658203125 \
        pitch_std=37.389366149902344 \
        +exp_manager.create_wandb_logger=true \
        +exp_manager.wandb_logger_kwargs.name=${wandb_run_name} \
        +exp_manager.wandb_logger_kwargs.project=${wandb_project_name}
fi

if [ ${stage} -eq 4 ]; then
    echo "synthesize"
fi