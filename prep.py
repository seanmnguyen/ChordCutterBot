import os
import torch
import time
import requests
import tarfile
import librosa
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import noisereduce as nr
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from sklearn.decomposition import PCA
import re
import platform

def compute_and_save_chromas(audio_paths, output_dir, sr=None):
    """
    For each filepath in audio_paths:
      - compute chroma
      - save as .npy in output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    for audio_path in audio_paths:
        # 1) Load the audio
        audio_data, sr_ = librosa.load('datashare/' + audio_path, sr=sr)
        audio_data = librosa.util.normalize(audio_data)

        # 2) Compute chroma (Shape: (12, time_frames))
        chroma_features = librosa.feature.chroma_stft(y=audio_data, sr=sr_)
        chroma_features /= np.max(chroma_features)  # scale [0, 1]

        # 3) Save .npy
        #    e.g. if audio_path = "/path/to/audio/song.wav"
        #    you'll get output_dir + "song.npy"
        filename_stem = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(output_dir, filename_stem + ".npz")
        np.savez(out_path, chroma=chroma_features, sr=sr_)

    print("Precomputation complete!")

def main():
    print('Fetching filepaths from MusicBench... Please wait')
    ds = load_dataset("amaai-lab/MusicBench")
    train_df_master = ds['train'].to_pandas()
    test_df_master = ds['test'].to_pandas()
    train_df = train_df_master.copy()
    test_df = test_df_master.copy()

    print("Before you can run the model on the anything from the dataset, you must do the following:")
    print("\t1) Download the MusicBench.tar.gz file from the dataset linked in the README")
    print("\t2) Extract the tar file")
    print("\t3) Pre-compute the chromas. You will be prompted below to perform this step.")
    print("Note: All of these steps will take a while (possibly 1+ hr) and use ~25 GB of storage as the dataset is very large.\n")

    precompute = input("Would you like to precompute the chromas now? (y/n): ")

    if precompute.strip().lower() == 'y':
        print("Computing and writing train chromas...")
        compute_and_save_chromas(train_df['location'], "chroma_features")

        print("Computing and writing test chromas...")
        compute_and_save_chromas(test_df['location'], "chroma_features_test")

        print("Preprocessing complete! Enjoy ChordCutter!")
    else:
        print("byebye")


if __name__ == "__main__":
    main()