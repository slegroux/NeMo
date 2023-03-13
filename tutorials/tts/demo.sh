#!/usr/bin/env bash
# Demo:

# Rishav
# Text 1: especially with the changes we made to the data collection pipeline
# Text 2: because at the end of the day that's what matters,

# Mayada:
# Text 1: But it doesn't have to be an actual meeting. It can just be
# Text 2: And that's showing a lot of good progress.

# CHECKPOINTS
exp_dir=rishav_600
# exp_dir=mayada_5min
specgen_ckpt_dir=${exp_dir}_finetune_fastpitch
vocoder_ckpt_dir=${exp_dir}_finetune_hifigan
# PRETRAINED
# pretrained_vocoder="tts_hifigan"
# pretrained_vocoder="tts_en_hifitts_hifigan_ft_fastpitch"
# sr=44100
sr=22050
# MODULATIONS
speed_factor=1.2
pitch_factor=1.0
pitch_scale=0.0
# INPUT
synth_text="especially with the changes we made to the data collection pipeline"
# synth_text="because at the end of the day that's what matters,"
# synth_text="But it doesn't have to be an actual meeting. It can just be"
# synth_text="And that's showing a lot of good progress."
text_underscore=$(echo ${synth_text}|sed 's| |_|g')

# --vocoder_ckpt_dir ${vocoder_ckpt_dir} \
# --vocoder ${pretrained_vocoder} \
python fastpitch_synth.py \
    --specgen_ckpt_dir ${specgen_ckpt_dir} \
    --vocoder_ckpt_dir ${vocoder_ckpt_dir} \
    --speed_factor ${speed_factor} \
    --pitch_factor ${pitch_factor} \
    --pitch_scale ${pitch_scale} \
    --input_text "${synth_text}" \
    --output_audio "${exp_dir}_${pretrained_vocoder}_${text_underscore}-sp${speed_factor}-pf${pitch_factor}-ps${pitch_scale}.wav" \
    --sr ${sr}