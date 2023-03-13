#!/usr/bin/env bash

exp_dir=kareem_5min
specgen_ckpt_dir=${exp_dir}_finetune_fastpitch
vocoder_ckpt_dir=${exp_dir}_finetune_hifigan
speed_factor=1.0
pitch_factor=1.0
pitch_scale=0.0

synth_text="This is an example of an adapted speaker."
text_underscore=$(echo ${synth_text}|sed 's| |_|g')

function usage {
        echo "Usage: $(basename $0) [-sft]" 2>&1
        echo 'Generate a random password.'
        echo '   -s SPEED           '
        echo '   -f PITCH_FACTOR    '
        echo '   -t PITCH_TRANSPOSE '
        exit 1
}

optstring="s:f:t:"
while getopts ${optstring} arg; do
  case "${arg}" in
    s)
        echo 'speed'
        speed_factor="${OPTARG}"
    ;;
    f)
        echo "pitch factor"
        pitch_factor="${OPTARG}"
    ;;
    t) 
        echo "pitch transpose"
        pitch_scale="${OPTARG}"
    ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      echo
      usage
      ;;
  esac
done


python fastpitch_synth.py \
    --specgen_ckpt_dir ${specgen_ckpt_dir} \
    --vocoder_ckpt_dir ${vocoder_ckpt_dir} \
    --speed_factor ${speed_factor} \
    --pitch_factor ${pitch_factor} \
    --pitch_scale ${pitch_scale} \
    --input_text "${synth_text}" \
    --output_audio "${exp_dir}_${text_underscore}-sp${speed_factor}-pf${pitch_factor}-ps${pitch_scale}.wav" \
    --sr 22050