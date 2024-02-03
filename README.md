# Vision Language Model Data Generation


Can we distill a tiny little VLM from an ensemble of the SOTA OSS models?

- generate dataset
    - llm creates prompt from human prompt
    - sdxl generates image from prompt
    - vlm1, vlm2, etc generate text from image
- create tiny vlm that can run on rpi (robot)
    - train from scratch on generated dataset
- run tiny vlm on rpi (robot)
    - further finetune on robot generated data

### Dataset

"A photograph of a person doing X"
"Describe the image" "What do you see?" "What is this person doing"
use llm to augment prompt, make varieties
need something easy to test with a camera that makes compelling demo
local webcam demo script

### Tiny VLM

LLM - Docker image with PyTorch for training
https://github.com/state-spaces/mamba
https://github.com/lucidrains/MEGABYTE-pytorch

Vision Encoder
CLIP + DINO visual encoders as quantized as possible. Can these be distilled as well?
https://huggingface.co/openai/clip-vit-large-patch14-336
https://huggingface.co/facebook/dinov2-base

3 stage training pipeline
MoE split over 2x1080GPUs
Can the three stage pipeline be short enough to fit inside a generative code evolution round meta learning loop?

generate dataset for "captioning" llm prompt > sdxl image > vlm text
launch docker with pytorch, huggingface
    pre-extract vision features, text response, text prompt
        frozen pre-trained vision encoder
        frozen pre-trained llm
per player
    generate code for connector mlp
    lint code
kill docker
per player
    launch docker mounted with pre-extracted features
    train connector mlp for X epochs
    train llm and connector mlp for Y epochs
    optional moe over vision encoders
    save result
compare results with llm