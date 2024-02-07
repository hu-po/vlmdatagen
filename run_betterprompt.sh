LLM="gpt"
DATA_DIR="/home/oop/dev/data/seedpromptscapgen"
python3 gensdxl.py --data_dir "$DATA_DIR" 
--seed=0  \
--dataset_size=640  \
--llm="$LLM"  \
--num_prompts=24
python3 capllava.py --data_dir "$DATA_DIR"
--seed=0  \
--llm="$LLM"  \
--num_prompts=6  \

