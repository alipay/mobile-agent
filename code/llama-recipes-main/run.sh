
 torchrun --nnodes 1 \
         --nproc_per_node 1 llama-recipes-main/examples/finetuning.py \
         --enable_fsdp \
         --use_peft \
         --peft_method lora \
         --model_name /mntnlp/common_base_model/llama2-7b \
         --dataset aitw_dataset \
         --output_dir /mntnlp/tine/temp \
         --use_fast_kernels \
         --run_validation False \
         --batch_size_training 4 \
         --num_epochs 10 \
         --quantization False \
