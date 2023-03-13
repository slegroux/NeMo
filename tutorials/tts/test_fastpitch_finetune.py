home_path = "/home/syl20"
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

STAGE = 0

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

def get_best_ckpt_from_last_run(
        base_dir, 
        new_speaker_id, 
        duration_mins, 
        mixing_enabled, 
        original_speaker_id, 
        model_name="FastPitch"
    ):    
    mixing = "no_mixing" if not mixing_enabled else "mixing"
    d = f"{original_speaker_id}_to_{new_speaker_id}_{mixing}_{duration_mins}_mins"
    exp_dirs = list([i for i in (Path(base_dir) / d / model_name).iterdir() if i.is_dir()])
    last_exp_dir = sorted(exp_dirs)[-1]
    last_checkpoint_dir = last_exp_dir / "checkpoints"
    last_ckpt = list(last_checkpoint_dir.glob('*-last.ckpt'))
    if len(last_ckpt) == 0:
        raise ValueError(f"There is no last checkpoint in {last_checkpoint_dir}.")
    return str(last_ckpt[0])

def download_fastpitch():
    # DOWNLOAD PRETRAINED MODEL
    FastPitchModel.from_pretrained("tts_en_fastpitch")
    nemo_files = [p for p in Path(f"{home_path}/.cache/torch/NeMo/").glob("**/tts_en_fastpitch_align.nemo")]
    print(f"Copying {nemo_files[0]} to ./")
    Path("./tts_en_fastpitch_align.nemo").write_bytes(nemo_files[0].read_bytes())

def get_vocoder():
    # VOCODER
    vocoder = HifiGanModel.from_pretrained("tts_hifigan")
    vocoder = vocoder.eval().cuda()
    return vocoder

def get_fastpitch(checkpoint):
    # FASTPITCH FINE-TUNED ON 1 SPEAKER
    # last_ckpt = get_best_ckpt_from_last_run("./", new_speaker_id, duration_mins, mixing, original_speaker_id)
    # print(last_ckpt)
    # last_ckpt = './ljspeech_to_6097_no_mixing_5_mins/FastPitch/2022-06-16_20-27-51/checkpoints/FastPitch--v_loss=1.6996-epoch=199-last.ckpt'
    spec_model = FastPitchModel.load_from_checkpoint(checkpoint)
    spec_model.eval().cuda()
    return spec_model

def eval_speaker_synthesis(spec_model, vocoder):
    # SYNTHESIZE SPEAKER
    new_speaker_id = 6097
    duration_mins = 5
    mixing = False
    original_speaker_id = "ljspeech"
    speaker_id = None

    num_val = 2  # Number of validation samples
    val_records = []
    with open(f"{new_speaker_id}_manifest_dev_ns_all_local.json", "r") as f:
        for i, line in enumerate(f):
            val_records.append(json.loads(line))
            if len(val_records) >= num_val:
                break

    for val_record in val_records:
        print(f"SYNTHESIZED FOR -- Speaker: {new_speaker_id} | Dataset size: {duration_mins} mins | Mixing:{mixing} | Text: {val_record['text']}")
        spec, audio = infer(spec_model, vocoder, val_record['text'], speaker=speaker_id)
        torchaudio.save(f'{new_speaker_id}.wav', torch.Tensor(audio), 22050) # encoding="PCM_S", bits_per_sample=16)

def synthesize(spec_model, vocoder, text, filename):
    speaker_id = None
    spec, audio = infer(spec_model, vocoder, text, speaker=speaker_id)
    torchaudio.save(f'{filename}', torch.Tensor(audio), 22050)

def hifigan_data_prep(spec_model):
    # DATA PREP FOR HIFIGAN FINE-TUNING
    def load_wav(audio_file):
        with sf.SoundFile(audio_file, 'r') as f:
            samples = f.read(dtype='float32')
        return samples.transpose()

    # Get records from the training manifest
    manifest_path = "./6097_manifest_train_dur_5_mins_local.json"
    records = []
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            records.append(json.loads(line))

    beta_binomial_interpolator = BetaBinomialInterpolator()
    # FASTPITCH
    spec_gen_model = spec_model
    spec_gen_model.eval()
    device = spec_gen_model.device
    save_dir = Path("./6097_manifest_train_dur_5_mins_local_mels")
    save_dir.mkdir(exist_ok=True, parents=True)

    # Generate a spectrograms (we need to use ground truth alignment for correct matching between audio and mels)
    for i, r in enumerate(records):
        audio = load_wav(r["audio_filepath"])
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
            # save_path = save_dir / f"mel_{i}.npy"
            # np.save(save_path, spectrogram[0].to('cpu').numpy())
            save_path = save_dir / f"mel_{i}.pt"
            torch.save(spectrogram[0].to('cpu'),save_path)
            r["mel_filepath"] = str(save_path)



    hifigan_manifest_path = "hifigan_train_ft.json"
    with open(hifigan_manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

# TODO:
# Train on omar data and check audio quality


# vocoder = get_vocoder()
vocoder = HifiGanModel.load_from_checkpoint('/home/syl20/slgNM/tutorials/tts/hifigan_ft/HifiGan/2022-06-17_18-30-54/checkpoints/HifiGan--val_loss=0.4349-epoch=999-last.ckpt')
vocoder = vocoder.eval().cuda()
fastpitch_checkpoint = './ljspeech_to_6097_no_mixing_5_mins/FastPitch/2022-06-16_20-27-51/checkpoints/FastPitch--v_loss=1.6996-epoch=199-last.ckpt'
fastpitch = get_fastpitch(fastpitch_checkpoint)
# synthesize(fastpitch, vocoder, "This is a test. Hope it'll work out.", 'test.wav')
hifigan_data_prep(fastpitch)
