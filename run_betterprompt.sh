DATA_DIR="/home/oop/dev/data"
python3 gensdxl.py --data_dir "$DATA_DIR" 
--seed=0  \
--data_dir=None  \
--base_dir="/home/oop/dev/data"  \
--dataset_size=640  \
--dataset_split=0.8  \
--llm="gpt"  \
--num_prompts=64
python3 capllava.py 
