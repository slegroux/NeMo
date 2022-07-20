#!/usr/bin/env python
import logging
logging.disable(logging.CRITICAL)
import argparse
import json
from pathlib import Path
import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
import torchaudio
from IPython import embed
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from scipy.io.wavfile import write
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.tts.torch.helpers import BetaBinomialInterpolator
import librosa
logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)

def get_best_ckpt_from_last_run(base_dir, checkpoints_dir, model_name='FastPitch'):
    exp_dirs = list([i for i in (Path(base_dir) / checkpoints_dir / model_name).iterdir() if i.is_dir()])
    last_exp_dir = sorted(exp_dirs)[-1]
    last_checkpoint_dir = last_exp_dir / "checkpoints"
    last_ckpt = list(last_checkpoint_dir.glob('*-last.ckpt'))
    if len(last_ckpt) == 0:
        raise ValueError(f"There is no last checkpoint in {last_checkpoint_dir}.")
    return str(last_ckpt[0])

def load_wav(audio_file, target_sr=None):
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read(dtype='float32')
        sample_rate = f.samplerate
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
    return samples.transpose()


def hifigan_data_prep(manifest_path, mels_dir, spec_gen_model, hifigan_manifest_path, target_sr=None):
    # Get list of records from manifest
    records = []
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            records.append(json.loads(line))

    beta_binomial_interpolator = BetaBinomialInterpolator()
    spec_gen_model.eval()
    device = spec_gen_model.device

    save_dir = Path(mels_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Generate a spectrograms (we need to use ground truth alignment for correct matching between audio and mels)
    for i, r in enumerate(records):
        audio = load_wav(r["audio_filepath"], target_sr=target_sr)
        audio = torch.from_numpy(audio).unsqueeze(0).to(device)
        audio_len = torch.tensor(audio.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        
        if spec_gen_model.fastpitch.speaker_emb is not None and "speaker" in r:
            speaker = torch.tensor([r['speaker']]).to(device)
        else:
            speaker = None
        
        with torch.no_grad():
            if "normalized_text" in r:
                text = spec_gen_model.parse(r["normalized_text"], normalize=False)
            else:
                text = spec_gen_model.parse(r['text'])
            
            text_len = torch.tensor(text.shape[-1], dtype=torch.long, device=device).unsqueeze(0)
            spect, spect_len = spec_gen_model.preprocessor(input_signal=audio, length=audio_len)

            attn_prior = torch.from_numpy(
                beta_binomial_interpolator(spect_len.item(), text_len.item())
            ).unsqueeze(0).to(text.device)

            spectrogram = spec_gen_model.forward(
                text=text, 
                input_lens=text_len, 
                spec=spect, 
                mel_lens=spect_len, 
                attn_prior=attn_prior,
                speaker=speaker,
            )[0]
            save_path = save_dir / f"mel_{i}.npy"
            np.save(save_path, spectrogram[0].to('cpu').numpy())
            # save_path = save_dir / f"mel_{i}.pt"
            # torch.save(spectrogram[0].to('cpu'), save_path)
            r["mel_filepath"] = str(save_path)

    with open(hifigan_manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path")
    parser.add_argument("--mels_dir")
    parser.add_argument("--specgen_ckpt_dir")
    parser.add_argument("--hifigan_manifest_path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    last_ckpt = get_best_ckpt_from_last_run("./", args.specgen_ckpt_dir)
    print(last_ckpt)
    
    spec_generator = FastPitchModel.load_from_checkpoint(last_ckpt)
    spec_generator.eval().cuda()
    hifigan_data_prep(args.manifest_path, args.mels_dir, spec_generator, args.hifigan_manifest_path)
