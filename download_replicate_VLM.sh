# Commands to create the directories on your local machine
DATA_DIR="$DATA_DIR"
RUN_ID="your_run_id" # Replace "your_run_id" with the desired value

mkdir -p "$DATA_DIR/clip-vit-large-patch14-336"
mkdir -p "$DATA_DIR/llava-v1.6-mistral-7b"
docker run -it -p 5000:5000 --gpus=all r8.im/yorickvp/llava-v1.6-mistral-7b@sha256:4798da673efa7bc088aa046c2d5382d0c8b4fad971c828c3740d44feb7cbb471

# Commands to copy CLIP model files
docker cp $RUN_ID:/src/openai/clip-vit-large-patch14-336/config.json "$DATA_DIR/clip-vit-large-patch14-336/config.json"
docker cp $RUN_ID:/src/openai/clip-vit-large-patch14-336/preprocessor_config.json "$DATA_DIR/clip-vit-large-patch14-336/preprocessor_config.json"
docker cp $RUN_ID:/src/openai/clip-vit-large-patch14-336/pytorch_model.bin "$DATA_DIR/clip-vit-large-patch14-336/pytorch_model.bin"

# Commands to copy LLaMA model files
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/config.json "$DATA_DIR/llava-v1.6-mistral-7b/config.json"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/generation_config.json "$DATA_DIR/llava-v1.6-mistral-7b/generation_config.json"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/model-00001-of-00004.safetensors "$DATA_DIR/llava-v1.6-mistral-7b/model-00001-of-00004.safetensors"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/model-00002-of-00004.safetensors "$DATA_DIR/llava-v1.6-mistral-7b/model-00002-of-00004.safetensors"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/model-00003-of-00004.safetensors "$DATA_DIR/llava-v1.6-mistral-7b/model-00003-of-00004.safetensors"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/model-00004-of-00004.safetensors "$DATA_DIR/llava-v1.6-mistral-7b/model-00004-of-00004.safetensors"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/model.safetensors.index.json "$DATA_DIR/llava-v1.6-mistral-7b/model.safetensors.index.json"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/special_tokens_map.json "$DATA_DIR/llava-v1.6-mistral-7b/special_tokens_map.json"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/tokenizer.model "$DATA_DIR/llava-v1.6-mistral-7b/tokenizer.model"
docker cp $RUN_ID:/src/liuhaotian/llava-v1.6-mistral-7b/tokenizer_config.json "$DATA_DIR/llava-v1.6-mistral-7b/tokenizer_config.json"
