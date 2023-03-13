#!/usr/bin/env python

from nemo_text_processing.text_normalization.normalize import Normalizer

from nemo.collections.tts.torch.data import TTSDataset
from nemo.collections.tts.torch.g2ps import EnglishG2p
from nemo.collections.tts.torch.tts_tokenizers import EnglishCharsTokenizer, EnglishPhonemesTokenizer

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
