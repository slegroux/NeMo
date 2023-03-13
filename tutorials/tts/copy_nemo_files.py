import json
import os
from pathlib import Path

import IPython.display as ipd
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from nemo.collections.tts.models import FastPitchModel

FastPitchModel.from_pretrained("tts_en_fastpitch")

HOME_PATH="/home/syl20/"
nemo_files = [p for p in Path(f"{HOME_PATH}/.cache/torch/NeMo/").glob("**/tts_en_fastpitch_align.nemo")]
print(f"Copying {nemo_files[0]} to ./")
Path("./tts_en_fastpitch_align.nemo").write_bytes(nemo_files[0].read_bytes())
