import os
import glob
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--dataset_path", type=str, default=None, help='dataset path')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    datapath = args.dataset_path
    files = glob.glob(os.path.join(datapath, 'phonemes', '*.txt'))
    savepath = os.path.join(datapath, 'vocab.txt')
    phn_vocab = []
    
    for f in tqdm(files):
        with open(f, 'r') as fi:
            data = fi.readlines()
        for x in data:
            x = x.split("\n")[0] if "\n" in x else x
            phn_vocab.append(x.split(" "))
    phn_vocab = set(phn_vocab)
    print(len(phn_vocab))
    with open(savepath, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")