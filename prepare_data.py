""" List audio files and transcripts, then create Pytorch-NLP tokenizer.
"""
import torch
import os
import glob
import argparse
import pandas as pd
import data_utils
from torchnlp.encoders.text import StaticTokenizerEncoder


# Core test set 24 speakers
SPEAKERS_TEST = [
    'MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', 'FPAS0', 'MJMP0', 'MLNT0', 'FPKT0', 'MLLL0', 'MTLS0', 'FJLM0',
    'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0', 'MJDH0', 'FMGD0', 'MGRT0', 'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0']


def read_phonemes(audio_file):
    """
    Given an audio file path, return its labelling.

    Args:
        audio_file (string): Audio file path.

    Returns:
        phonemes (string): A sequence of phonemes for this audio file.
    """
    phn_file = audio_file[:-4] + '.PHN'
    with open(phn_file) as f:
        phonemes = f.readlines()
    phonemes = ' '.join([l.strip().split()[-1] for l in phonemes])
    return phonemes


def process_dataset(root, split):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.

    Args:
        root (string): Directory of TIMIT.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    if split == 'train':
        audio_files = glob.glob(os.path.join(root, "TRAIN/**/*.WAV"), recursive=True)
    else:
        audio_files = glob.glob(os.path.join(root, "TEST/**/*.WAV"), recursive=True)
        if split == 'dev':
            audio_files = [p for p in audio_files if p.split('/')[-2] not in SPEAKERS_TEST]
        else:
            audio_files = [p for p in audio_files if p.split('/')[-2] in SPEAKERS_TEST]
    # Remove all 'SA' records.
    audio_files = [p for p in audio_files if 'SA' not in os.path.basename(p)]
    transcripts = [read_phonemes(p) for p in audio_files]

    fname = '%s.csv'%split.upper()
    with open(fname, 'w') as f:
        f.write("audio,transcript\n")
        for (x, y) in zip(audio_files, transcripts):
            f.write("%s,%s\n" % (x, y))
    print ("%s is created." % fname)


def create_tokenizer():
    """
    Create and save Pytorch-NLP tokenizer.

    Args:
        root (string): Directory of TIMIT.
    """
    transcripts = pd.read_csv('TRAIN.csv')['transcript']
    tokenizer = StaticTokenizerEncoder(transcripts,
                                       append_sos=True,
                                       append_eos=True,
                                       tokenize=data_utils.encode_fn,
                                       detokenize=data_utils.decode_fn)
    torch.save(tokenizer, 'tokenizer.pth')


def main():
    parser = argparse.ArgumentParser(description="Make lists of audio files and transcripts, and create tokenizer.")
    parser.add_argument('--root', default="data/lisa/data/timit/raw/TIMIT", type=str, help="Directory of TIMIT.")
    args = parser.parse_args()

    process_dataset(args.root, 'train')
    process_dataset(args.root, 'dev')
    process_dataset(args.root, 'test')

    create_tokenizer()
    print ("Data preparation is complete !")


if __name__ == '__main__':
    main()
