import os
import numpy as np
import pyworld as pw
import librosa
import soundfile as sf
from scipy.interpolate import interp1d
import random
from scipy.signal import resample
import multiprocessing
from tqdm import tqdm
import pandas as pd
import argparse

# Modify the spectral envelope
def modify_spectral_envelope(sp, fs):
    ratio = random.uniform(0.8, 1.25)
    fft_size = (sp.shape[1] - 1) * 2
    f0 = np.fft.fftfreq(fft_size, 1.0 / fs)
    f0[f0 < 0] += fs
    f0 *= ratio
    sp_modified = np.zeros_like(sp)
    for i in range(sp.shape[0]):
        sp_modified[i] = np.interp(f0[:sp.shape[1]], f0[:sp.shape[1]] * ratio, sp[i])
    return sp_modified

def process_one(i, splits, rdata, savedir):
    for split in tqdm(splits):
        if os.path.exists(rdata['wav'][split]):
            audiopath = rdata['wav'][split]
        else:
            audiopath = rdata['wav'][split].replace('/apdcephfs_cq2', '/apdcephfs_cq2_1297902')

        save_dir = os.path.join(savedir, rdata['segment_id'][split]+'.wav')
        if not os.path.exists(save_dir):
            # Load the audio file
            x, fs = librosa.load(audiopath, mono=True, sr=16000)
            x = x.astype(np.float64)
            # Extract the F0 (fundamental frequency), spectral envelope, and aperiodicity
            _f0, t = pw.harvest(x, fs)
            sp = pw.cheaptrick(x, _f0, t, fs)
            ap = pw.d4c(x, _f0, t, fs)
            
            # Apply the spectral envelope modification
            sp_modified = modify_spectral_envelope(sp, fs)
            
            # Synthesize the audio using the modified spectral envelope
            y_modified = pw.synthesize(_f0, sp_modified, ap, fs)
            y_modified = y_modified.astype(np.float64)
            # Save the result
            sf.write(save_dir, y_modified, fs)

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def parse_args():
    parser = argparse.ArgumentParser(description="encode the dataset using encodec model")
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    save_dir = args.save_dir
    dataset_dir = args.dataset_dir
    
    os.makedirs(save_dir, exist_ok=True)
    for split in splits:
        cmds = []
        rdata = pd.read_json(path_or_buf=os.path.join(dataset_dir, 'trans', split+'.json'), lines=True)
        tmp = list(np.arange(len(rdata)))
        random.shuffle(tmp)
        split_parts = split_list(tmp, 88)
        for i in range(len(split_parts)):
            cmds.append((i, split_parts[i], rdata, save_dir))         
        with multiprocessing.Pool(processes=88) as pool:
            pool.starmap(process_one, cmds)