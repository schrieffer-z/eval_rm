export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=7

ckpts+=(checkpoint-375)
ckpts+=(checkpoint-350)
ckpts+=(checkpoint-325)
ckpts+=(checkpoint-300)
ckpts+=(checkpoint-275)
ckpts+=(checkpoint-250)
ckpts+=(checkpoint-225)
ckpts+=(checkpoint-200)
ckpts+=(checkpoint-175)
ckpts+=(checkpoint-150)
ckpts+=(checkpoint-125)
ckpts+=(checkpoint-100)
ckpts+=(checkpoint-75)
ckpts+=(checkpoint-50)
ckpts+=(checkpoint-25)


for ckpt in ${ckpts[@]};do
    accelerate launch \
        --main_process_port 29193 \
        --config_file ./scripts/accelerate_configs/deepspeed_zero0_reward_bench_1gpu.yaml \
        ./scripts/run_rm.py \
        --batch_size=4 \
        --model /mnt/finder/lisihang/xAI-RLHF/Shuyi/sae4rm/models/Llama-3.1-8B-Instruct_token_Latent32768_Layer16_K192_10M-SAE4RM-Sparsified/last_checkpoint \
        --tokenizer /mnt/finder/lisihang/xAI-RLHF/Shuyi/sae4rm/models/Llama-3.1-8B-Instruct_token_Latent32768_Layer16_K192_10M-SAE4RM-Sparsified/last_checkpoint \
        --sae4rm_base_model llama \
        --sae_path Llama-3.1-8B-Instruct_token_Latent32768_Layer16_K192_10M \
        --attn_implementation flash_attention_2 \
        --sae4rm_use_topk 
done