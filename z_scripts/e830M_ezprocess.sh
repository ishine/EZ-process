#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8

dataset=English
mkdir -p ./logs/${dataset}

exp_root="/apdcephfs_cq10/share_1603164/user/helinhwang/EZ-process/exp_results"
exp_name=e830M_ezprocess
dataset_dir="/apdcephfs_cq10/share_1603164/data/Audio_Edit/VoiceCraft_English/English" # xs if you only extracted xs in previous step
encodec_codes_folder_name="encodec_16khz_4codebooks"
encodec_aug_folder_name="encodec_16khz_4codebooks_prosody_speakingrate_augmented"
prosody_timbre_folder_name="codes_facodec"

export CUDA_LAUNCH_BLOCKING=1 # for debugging
export TORCH_USE_CUDA_DSA=1

torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:41977 --nproc_per_node=${WORLD_SIZE} \
../main_ezprocess.py \
--resume \
--reduced_eog 0 \
--drop_long 1 \
--eos 2051 \
--sos 2052 \
--mts 2053 \
--predict_mask_token 1 \
--predict_all 0 \
--n_special 5 \
--pad_x 0 \
--codebook_weight "[5,1,0.5,0.1]" \
--encodec_sr 50 \
--num_steps 1000000000 \
--lr 0.05 \
--warmup_fraction 0.01 \
--optimizer_name "ScaledAdam" \
--pseudo_epoch_size 3000 \
--reduce_lr_start_step 3000 \
--reduce_lr_start_epoch 4 \
--clipping_update_period 1000 \
--d_model 2048 \
--audio_embedding_dim 2048 \
--nhead 16 \
--num_decoder_layers 16 \
--max_num_tokens 100000 \
--gradient_accumulation_steps 100 \
--val_max_num_tokens 6000 \
--num_buckets 6 \
--audio_max_length 15 \
--audio_min_length 1 \
--prosody_max_length 3 \
--text_max_length 400 \
--text_min_length 2 \
--mask_len_min 1 \
--mask_len_max 600 \
--tb_write_every_n_steps 10 \
--print_every_n_steps 400 \
--val_every_n_steps 400 \
--text_vocab_size 100 \
--text_pad_token 100 \
--phn_folder_name "phonemes" \
--manifest_name "manifest_aug" \
--encodec_folder_name ${encodec_codes_folder_name} \
--encodec_aug_folder_name ${encodec_aug_folder_name} \
--prosody_timbre_folder_name ${prosody_timbre_folder_name} \
--spk_emb 256 \
--audio_vocab_size 2048 \
--prosody_vocab_size 1024 \
--prosody_pad_token 1024 \
--empty_token 2048 \
--eog 2049 \
--audio_pad_token 2050 \
--n_codebooks 4 \
--max_n_spans 3 \
--tts_enhanced 1 \
--shuffle_mask_embedding 0 \
--mask_sample_dist uniform \
--max_mask_portion 0.9 \
--min_gap 5 \
--train_gap 10 \
--num_workers 32 \
--dynamic_batching 1 \
--dataset $dataset \
--exp_dir "${exp_root}/${dataset}/${exp_name}" \
--dataset_dir ${dataset_dir}
