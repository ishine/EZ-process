# import libs
# if this throws an error, something went wrong installing dependencies or changing the kernel above!
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USER"] = "root" # TODO change this to your username

import shutil
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
import torchaudio
import torchaudio.transforms as transforms
from edit_utils import parse_edit
from inference_scale import get_mask_interval
from inference_scale import inference_one_sample
import time
from tqdm import tqdm

# hyperparameters for inference
left_margin = 0.08
right_margin = 0.08
sub_amount = 0.01
codec_audio_sr = 16000
codec_sr = 50
top_k = 0
top_p = 0.8
temperature = 1
kvcache = 1
# adjust the below three arguments if the generation is not as good
seed = 1 # random seed magic
silence_tokens = [1388,1898,131] # if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
stop_repetition = -1 # -1 means do not adjust prob of silence tokens. if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4
sample_batch_size = 5 # what this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")
# load model, tokenizer, and other necessary files
voicecraft_name="best_bundle.pth" # or gigaHalfLibri330M_TTSEnhanced_max16s.pth, giga830M.pth

# # the old way of loading the model
from models import voicecraft
filepath = os.path.join('/apdcephfs_cq10_1603164/share_1603164/user/helinhwang/VoiceCraft/exp_results/Chinese/e830M_zh_enh_25kv2/', voicecraft_name)
ckpt = torch.load(filepath, map_location="cpu")
model = voicecraft.VoiceCraft(ckpt["config"])
model.load_state_dict(ckpt["model"])
config = vars(model.args)
phn2num = ckpt["phn2num"]
model.to(device)
model.eval()
encodec_fn = "/apdcephfs_cq10_1603164/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/VoiceCraft/encodec_4cb2048_giga.th"
audio_tokenizer = AudioTokenizer(signature=encodec_fn) # will also put the neural codec model on gpu
text_tokenizer = TextTokenizer(backend="espeak")


def main(orig_audio, orig_transcript, target_transcript, target_transcript_, temp_folder, savename, savetag=1):
    start_time = time.time()
    # move the audio and transcript to temp folder
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp {orig_audio} {temp_folder}")
    
    filename = os.path.splitext(orig_audio.split("/")[-1])[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(' '.join(orig_transcript))

    # resampling
    import librosa
    import soundfile as sf
    audio, sr = librosa.load(os.path.join(temp_folder, filename+'.wav'), sr=16000)
    sf.write(os.path.join(temp_folder, filename+'.wav'), audio, 16000)
        
    # run MFA to get the alignment
    align_temp = f"{temp_folder}/mfa_alignments"
    os.makedirs(align_temp, exist_ok=True)
    # os.system(f"mfa align --overwrite -j 1 --output_format csv {temp_folder} /apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/mandarin_china_mfa.dict /apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/mandarin_mfa.zip {align_temp} --clean")
    # os.system(f"mfa align -j 1 --output_format csv {temp_folder} /apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/mandarin_china_mfa.dict /apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/mandarin_mfa.zip {align_temp} --beam 1000 --retry_beam 2000")

    audio_fn = f"{temp_folder}/{filename}.wav"
    transcript_fn = f"{temp_folder}/{filename}.txt"
    align_fn = f"{align_temp}/{filename}.csv"

    # run the script to turn user input to the format that the model can take
    operations, orig_spans = parse_edit(orig_transcript, target_transcript)
    print(operations)
    print("orig_spans: ", orig_spans)
    
    if len(orig_spans) > 3:
        raise RuntimeError("Current model only supports maximum 3 editings")
        
    starting_intervals = []
    ending_intervals = []
    for orig_span in orig_spans:
        start, end = get_mask_interval(align_fn, orig_span)
        starting_intervals.append(start)
        ending_intervals.append(end)

    print("intervals: ", starting_intervals, ending_intervals)

    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate
    
    def resolve_overlap(starting_intervals, ending_intervals, audio_dur, codec_sr, left_margin, right_margin, sub_amount):
        while True:
            morphed_span = [(max(start - left_margin, 1/codec_sr), min(end + right_margin, audio_dur))
                            for start, end in zip(starting_intervals, ending_intervals)] # in seconds
            mask_interval = [[round(span[0]*codec_sr), round(span[1]*codec_sr)] for span in morphed_span]
            # Check for overlap
            overlapping = any(a[1] >= b[0] for a, b in zip(mask_interval, mask_interval[1:]))
            if not overlapping:
                break
            
            # Reduce margins
            left_margin -= sub_amount
            right_margin -= sub_amount
        
        return mask_interval
    
    # span in codec frames
    mask_interval = resolve_overlap(starting_intervals, ending_intervals, audio_dur, codec_sr, left_margin, right_margin, sub_amount)
    mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now
    print(mask_interval)
    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens}
    new_audios = []
    for num in tqdm(range(sample_batch_size)):
        seed_everything(seed+num)
        orig_audio, new_audio = inference_one_sample(model, Namespace(**config), phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, mask_interval, device, decode_config)
        # save segments for comparison
        orig_audio, new_audio = orig_audio[0].cpu(), new_audio[0].cpu()
        new_audios.append(new_audio)


    for num in range(sample_batch_size):
        # print(new_audios[num].shape)
        if new_audios[num].shape[0] < new_audio.shape[0]:
            new_audio = new_audios[num]
            
    output_dir = "/apdcephfs_cq10_1603164/share_1603164/user/helinhwang/VoiceCraft/demo/generated_se"
    os.makedirs(output_dir, exist_ok=True)
    save_fn_new = f"{output_dir}/{savename}_new_seed{seed}_{str(savetag)}.wav"
    torchaudio.save(save_fn_new, new_audio, codec_audio_sr)
    
    save_fn_orig = f"{output_dir}/{savename}_orig.wav"
    if not os.path.isfile(save_fn_orig):
        orig_audio, orig_sr = torchaudio.load(audio_fn)
        if orig_sr != codec_audio_sr:
            orig_audio = torchaudio.transforms.Resample(orig_sr, codec_audio_sr)(orig_audio)
        torchaudio.save(save_fn_orig, orig_audio, codec_audio_sr)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Running time: {elapsed_time:.4f} s")

# orig_audio = "/apdcephfs_cq10_1603164/share_1603164/user/helinhwang/VoiceCraft/demo/pony3.wav"
# orig_transcript =    "能够更有效率地结合给用户提升更多的这种体验也包括他的这个他的后台的效率提升等等我相信这些额额业界的解决方案应该说是"
# target_transcript =  "能够更有效率地结合给用户提升更多的体验也包括后台的效率提升等等我相信这些业界的解决方案应该说是"
# target_transcript_ = "能够更有效率地结合给用户提升更多的体验也包括后台的效率提升等等我相信这些业界的解决方案应该说是"
# temp_folder = "/apdcephfs_cq10_1603164/share_1603164/user/helinhwang/VoiceCraft/demo/temp"
# main(orig_audio, orig_transcript, target_transcript, target_transcript_, temp_folder, savename='pony3',savetag=1)
# # shutil.rmtree(temp_folder)