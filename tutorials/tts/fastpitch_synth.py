#!/usr/bin/env python
# 06/2022 <sylvain.legroux@gmail.com>

import argparse
from pathlib import Path
import logging
logging.disable(logging.CRITICAL)
import soundfile as sf
import torch
from nemo.collections.tts.helpers.helpers import regulate_len
from nemo.collections.tts.models import FastPitchModel, HifiGanModel, UnivNetModel
logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)

def infer(spec_gen_model, vocoder_model, str_input, speaker=None):
    """
    Synthesizes spectrogram and audio from a text string given a spectrogram synthesis and vocoder model.
    
    Args:
        spec_gen_model: Spectrogram generator model (FastPitch in our case)
        vocoder_model: Vocoder model (HiFiGAN in our case)
        str_input: Text input for the synthesis
        speaker: Speaker ID
    
    Returns:
        spectrogram and waveform of the synthesized audio.
    """
    with torch.no_grad():
        parsed = spec_gen_model.parse(str_input)
        if speaker is not None:
            speaker = torch.tensor([speaker]).long().to(device=spec_gen_model.device)
        spectrogram = spec_gen_model.generate_spectrogram(tokens=parsed, speaker=speaker)
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
        
    if spectrogram is not None:
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.to('cpu').numpy()
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
    return spectrogram, audio

def get_best_ckpt_from_last_run(base_dir, checkpoints_dir, model_name='FastPitch'):

    exp_dirs = list([i for i in (Path(base_dir) / checkpoints_dir / model_name).iterdir() if i.is_dir()])
    last_exp_dir = sorted(exp_dirs)[-1]
    last_checkpoint_dir = last_exp_dir / "checkpoints"
    last_ckpt = list(last_checkpoint_dir.glob('*-last.ckpt'))

    if len(last_ckpt) == 0:
        raise ValueError(f"There is no last checkpoint in {last_checkpoint_dir}.")
    
    return str(last_ckpt[0])

def synthesize_speech(spec_generator, vocoder, text, pitch_factor=1.0, pitch_scale=0.0, speed_factor=1.0, speaker=1, durs=None, pitch=None):
    speaker = torch.IntTensor(speaker).cuda()
    with torch.no_grad():
        parsed = spec_generator.parse(text)
        # spectrogram = spec_generator.generate_spectrogram(tokens=parsed, speaker=speaker)
        spectrogram, _, durs_pred, _, pitch_pred, *_ = spec_generator(text=parsed, durs=durs, pitch=pitch, speaker=speaker, pace=1.0)
        new_pitch = pitch_pred * pitch_factor + pitch_scale
        spectrogram, _, durs_pred, _, pitch_pred, *_ = spec_generator(text=parsed, durs=durs, pitch=new_pitch, speaker=speaker, pace=speed_factor)
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram) #.to('cpu')
    return audio

def get_args():
    parser = argparse.ArgumentParser()
    # tts_en_fastpitch_multispeaker
    parser.add_argument("--spectrogram_generator", type=str, default="tts_en_fastpitch_multispeaker.nemo")
    parser.add_argument("--specgen_ckpt_dir")
    parser.add_argument("--vocoder_ckpt_dir")
    # tts_en_hifitts_hifigan_ft_fastpitch
    # "tts_en_libritts_univnet"
    parser.add_argument("--vocoder", type=str, default="tts_en_hifitts_hifigan_ft_fastpitch")
    parser.add_argument("--input_text", type=str, default="One, two. This is a test.")
    parser.add_argument("--output_audio", default='test.wav')
    parser.add_argument("--speaker", type=int, default=1)
    parser.add_argument("--pitch_factor", type=float, default=1.0)
    parser.add_argument("--pitch_scale", type=float, default=0.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--sr", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # MODELS
    # spec_generator = FastPitchModel.from_pretrained(args.spectrogram_generator, override_config_path=None)
    # spec_generator = FastPitchModel.restore_from(args.spectrogram_generator, override_config_path=None)
    if args.specgen_ckpt_dir:
        last_ckpt = get_best_ckpt_from_last_run("./", args.specgen_ckpt_dir, model_name='FastPitch')
        spec_generator = FastPitchModel.load_from_checkpoint(last_ckpt)
    else:
        spec_generator = FastPitchModel.from_pretrained(args.spectrogram_generator, override_config_path=None)
    spec_generator.eval().cuda()

    # vocoder = UnivNetModel.from_pretrained(model_name=args.vocoder)
    # vocoder = HifiGanModel.restore_from(args.vocoder)
    if args.vocoder_ckpt_dir:
        last_ckpt = get_best_ckpt_from_last_run("./", args.vocoder_ckpt_dir, model_name='HifiGan')
        vocoder = HifiGanModel.load_from_checkpoint(last_ckpt)
    else:
        vocoder = HifiGanModel.from_pretrained(args.vocoder)
    vocoder.eval().cuda()
    audio = synthesize_speech(spec_generator, vocoder, args.input_text, pitch_scale=args.pitch_scale, pitch_factor=args.pitch_factor, speed_factor=args.speed_factor, speaker=args.speaker)
    sf.write(args.output_audio, audio[0].detach().cpu().numpy(), args.sr)
    # spectrogram, audio = infer(spec_generator, vocoder, args.input_text, speaker=None)
    # sf.write(args.output_audio, audio[0], args.sr)
