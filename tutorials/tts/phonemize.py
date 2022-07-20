import argparse
import json

from phonemizer.backend import EspeakBackend


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--val')
    parser.add_argument('--lang')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    backend = EspeakBackend(args.lang)

    for input_manifest_filepath in [args.train, args.test, args.val]:
        output_manifest_filepath = input_manifest_filepath+"_phonemes"
        records = []
        n_text = []
        with open(input_manifest_filepath + ".json", "r") as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                records.append(d)
                n_text.append(d['normalized_text'])

        phonemized = backend.phonemize(n_text)

        new_records = []
        for i in range(len(records)):
            records[i]["is_phoneme"] = 0
            new_records.append(records[i])
            phoneme_record = records[i].copy()
            phoneme_record["normalized_text"] = phonemized[i]
            phoneme_record["is_phoneme"] = 1
            new_records.append(phoneme_record)

        with open(output_manifest_filepath + ".json", "w") as f:
            for r in new_records:
                f.write(json.dumps(r) + '\n')
