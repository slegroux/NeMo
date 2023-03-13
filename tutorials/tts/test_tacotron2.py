
#%% [markdown]
# ## Dataclass Config

#%%
# FEATURES
from dataclasses import dataclass
from typing import Optional

import hydra
import torch
import torchaudio
#%%
from matplotlib import pyplot as plt
# %%
# DATASET
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

# print(cfg.model.preprocessor)

@dataclass
class FilterbankConfig:
    dither:float = 0.0
    nfilt:int = 80
    frame_splicing:int = 1
    highfreq:float = 8000
    log:bool = True
    log_zero_guard_type:str = 'clamp'
    log_zero_guard_value:float = 1e-05
    lowfreq:float = 0
    mag_power:float = 1.0
    n_fft:int = 1024
    n_window_size:int = 1024
    n_window_stride:int = 256
    normalize:Optional[bool] = None
    pad_to:int = 16
    pad_value:float = -11.52
    preemph:Optional[bool] = None
    sample_rate:int = 16000
    window:str = 'hann'
    
filterbank_dict = FilterbankConfig()


f = OmegaConf.structured(FilterbankConfig())
print(OmegaConf.to_yaml(f))
print(type(f))

filter_bank = FilterbankFeatures(**f)
path = "audio/astralplane_03_leadbeater_0639.wav"
audio, sr = torchaudio.load(path)
print(audio.shape[1])

#%%
l = torch.IntTensor([audio.shape[1]])
mel = filter_bank(audio, l)
plt.imshow(mel[0].squeeze(), origin='lower')
plt.show()


cfg = OmegaConf.load('conf/tacotron2.yaml')

# validation_datasets=tests/data/asr/an4_val.json
# trainer.max_epochs=3 trainer.accelerator=null
# trainer.check_val_every_n_epoch=1
cfg.model.train_ds.dataset.sample_rate=16000
cfg.model.train_ds.dataset.manifest_filepath="tests/data/asr/an4_train.json"
labels = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z']
print(cfg.model.train_ds.dataset)
ds = hydra.utils.instantiate(cfg.model.train_ds.dataset, labels=labels)
dl = DataLoader(ds, batch_size=1)
dict_conf = OmegaConf.to_container(cfg.model.train_ds.dataset)
dict_conf.pop('_target_')

print(dict_conf)
ds = AudioToCharDataset(**dict_conf, labels=labels)
# %%
print(cfg.model.train_ds.dataloader_params)
dl = DataLoader(ds, **cfg.model.train_ds.dataloader_params)


# %%
