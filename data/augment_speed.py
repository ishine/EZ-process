import os
import numpy as np
import random
import multiprocessing
from tqdm import tqdm
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="encode the dataset using encodec model")
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    return parser.parse_args()

def process_one(i, splits, rdata, savedir):
    for split in tqdm(splits):
        audiopath = rdata[split]
        save_dir = os.path.join(savedir, audiopath.split("/")[-1])
        if not os.path.exists(save_dir):
            ratio = str(random.uniform(0.5, 2.0))
            os.system(f"ffmpeg -i {audiopath} -filter:a \"atempo={ratio}\" -b:a 320k {save_dir}")

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    args = parse_args()
    save_dir = args.save_dir
    dataset_dir = args.dataset_dir
    os.makedirs(save_dir, exist_ok=True)
    
    cmds = []
    rdata = glob.glob(os.path.join(dataset_dir, "*.wav"))
    random.shuffle(rdata)
    print(len(rdata))
    
    tmp = list(np.arange(len(rdata)))
    random.shuffle(tmp)
    split_parts = split_list(tmp, 88)
    for i in range(len(split_parts)):
        cmds.append((i, split_parts[i], rdata, save_dir))         
    with multiprocessing.Pool(processes=88) as pool:
        pool.starmap(process_one, cmds)