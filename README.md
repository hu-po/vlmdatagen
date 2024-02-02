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