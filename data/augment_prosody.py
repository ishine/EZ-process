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

# Modify the pitch (prosody)
def modify_pitch(f0, segment_length=20):
    original_length = len(f0)
    
    # Extend the length to match the segment length exactly
    if original_length % segment_length != 0:
        extended_length = ((original_length // segment_length) + 1) * segment_length
        f0 = np.pad(f0, (0, extended_length - original_length), 'edge')
    else:
        extended_length = original_length
    
    # Generate random pitch factors for each segment
    num_segments = extended_length // segment_length
    pitch_factors = np.random.uniform(0.7, 1.5, num_segments)

    # Create an array of indices for the segments
    segment_indices = np.arange(0, extended_length, segment_length)

    # Interpolate pitch factors to create a smooth transition
    interpolator = interp1d(segment_indices, pitch_factors, kind='linear', fill_value="extrapolate")
    smooth_pitch_factors = interpolator(np.arange(extended_length))

    # Apply the smooth pitch factors to the original (extended) F0
    modified_f0 = f0 * smooth_pitch_factors

    # Cut back to the original length
    modified_f0 = modified_f0[:original_length]
    return modified_f0

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
            
            
            # Apply prosody modifications
            modified_f0 = modify_pitch(_f0)
            # Synthesize the audio using the modified F0 and original spectral envelope and aperiodicity
            y_modified = pw.synthesize(modified_f0, sp, ap, fs)
            
            # Ensure the output array is of type 'double'
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
    
    splits = ['validation', 'test', 'train']
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