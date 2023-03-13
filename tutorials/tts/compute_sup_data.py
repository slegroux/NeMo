
#!/usr/bin/env python

import argparse

import nonechucks as nc
import torch
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm.notebook import tqdm

from nemo.collections.tts.torch.data import TTSDataset
from nemo.collections.tts.torch.g2ps import EnglishG2p
from nemo.collections.tts.torch.tts_tokenizers import EnglishCharsTokenizer, EnglishPhonemesTokenizer


def pre_calculate_supplementary_data(manifest_filepath, sup_data_path, sup_data_types, text_tokenizer, text_normalizer, text_normalizer_call_kwargs):
    # init train and val dataloaders
    ds = TTSDataset(
        manifest_filepath=manifest_filepath,
        sample_rate=16000,
        sup_data_path=sup_data_path,
        sup_data_types=sup_data_types,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        window="hann",
        n_mels=80,
        lowfreq=0,
        highfreq=8000,
        text_tokenizer=text_tokenizer,
        text_normalizer=text_normalizer,
        text_normalizer_call_kwargs=text_normalizer_call_kwargs

    ) 
    # ds = nc.SafeDataset(ds)
    # from IPython import embed; embed()
    # dl = nc.SafeDataLoader(ds, batch_size=1, collate_fn=ds._collate_fn, num_workers=1)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=ds._collate_fn, num_workers=1)

    # iteration over dataloaders
    pitch_mean, pitch_std, pitch_min, pitch_max = None, None, None, None

    pitch_list = []
    counter = 0
    # for batch in tqdm(dl, total=len(dl)):
    for batch in dl:
        try:
            tokens, tokens_lengths, audios, audio_lengths, attn_prior, pitches, pitches_lengths = batch
            pitch = pitches.squeeze(0)
            pitch_list.append(pitch[pitch != 0])
        except IOError:
            print("skip")
        print(f"{counter}")
        counter += 1 

    # if stage == "train":
    pitch_tensor = torch.cat(pitch_list)
    pitch_mean, pitch_std = pitch_tensor.mean().item(), pitch_tensor.std().item()
    pitch_min, pitch_max = pitch_tensor.min().item(), pitch_tensor.max().item()
            
    return pitch_mean, pitch_std, pitch_min, pitch_max

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sup_data_folder")
    parser.add_argument("--manifest_filepath")
    parser.add_argument("--whitelist")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Text normalizer
    text_normalizer = Normalizer(
        lang="en", 
        input_case="cased", 
        whitelist="whitelists/lj_speech.tsv"
    )
    text_normalizer_call_kwargs = {
        "punct_pre_process": True,
        "punct_post_process": True
    }
    # Text tokenizer
    text_tokenizer = EnglishCharsTokenizer()
    sup_data_types = ["align_prior_matrix", "pitch"]

    pitch_mean, pitch_std, pitch_min, pitch_max = pre_calculate_supplementary_data(
        args.manifest_filepath, args.sup_data_folder, \
        sup_data_types, text_tokenizer, text_normalizer, text_normalizer_call_kwargs
    )
