Dataset="humaneval"  # MBPP
Model="CodeLlama-7b-hf"
model_dir="/home/dongyh/LLMs/CodeLlama-7b-hf"
output_dir="outputs/${Dataset}"
suffix="ROCODE"


python main.py \
--arch ${Model} \
--model-dir ${model_dir} \
--dataset ${Dataset} \
--temperature 0.0 \
--topk 50 \
--topp 1 \
--num-samples 1 \
--decay-factor 0.9 \
--output-dir ${output_dir} \
--output-file-suffix ${suffix} \


python evaluate_generated_code.py \
--dataset ${Dataset} \
--input_path ${output_dir}/${Dataset}_${Model}_temp0.0_topp1.0_topk50_df0.9_samples1_${suffix}.jsonl \
--truncate \
--eval_standard \
--eval_ET