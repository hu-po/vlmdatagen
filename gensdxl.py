import argparse
import base64
import uuid
import os
import subprocess
import time
import random
import requests

from io import BytesIO
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--base_dir", type=str, default="/home/oop/dev/data")
parser.add_argument("--dataset_size", type=int, default=640)
parser.add_argument("--dataset_split", type=float, default=0.8)
parser.add_argument("--llm", type=str, default="gpt")
parser.add_argument("--num_prompts", type=int, default=64)
args = parser.parse_args()

if args.llm == "gpt":
    from textboi import import_gpt
    llm: callable = import_gpt()
elif args.llm == "rep":
    from textboi import import_rep
    llm: callable = import_rep()
else:
    raise ValueError(f"Unknown llm {args.llm}")

if args.data_dir is None:
    dataset_id = str(uuid.uuid4())[:6]
    print(f"No data directory specified, generating new dataset {dataset_id}")
    data_dir = os.path.join(args.base_dir, f"vlmgen.{dataset_id}")
else:
    data_dir = args.data_dir
os.makedirs(data_dir, exist_ok=True)
print(f"data directory at {data_dir}")
train_dir = os.path.join(data_dir, "train")
os.makedirs(train_dir, exist_ok=True)
print(f"train directory at {train_dir}")
test_dir = os.path.join(data_dir, "test")
os.makedirs(test_dir, exist_ok=True)
print(f"test directory at {test_dir}")

# Use llm to generate prompts
prompts = set()
while len(prompts) < args.num_prompts:
    reply = llm(
        """
You are a data generator.
You generate diverse data with a wide distribution.
Reply with comma separated lowercase single words.
Reply only with single words.
Return a comma separated list of words.
Do not use punctuation or capitalization.
Return at least 10 words.
        """,
        """
What are some food related words?
        """,
        1.6,
        64,
    )
    prompts.update(set([_.lower() for _ in reply.split(",")]))
prompts = random.sample(prompts, args.num_prompts)

# -------------- SDXL
docker_ps_process = subprocess.Popen(["docker", "ps"], stdout=subprocess.PIPE)
docker_ps_output, _ = docker_ps_process.communicate()
if "sdxl" in docker_ps_output.decode():
    print("Docker is already running.")
    sdxl_docker_proc = None
else:
    os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
    sdxl_docker_proc = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "-p",
            "5000:5000",
            "--gpus=all",
            "-v",
            "/home/oop/dev/data/sdxl/sdxl-cache:/src/sdxl-cache",
            "-v",
            "/home/oop/dev/data/sdxl/safety-cache:/src/safety-cache",
            "-v",
            "/home/oop/dev/data/sdxl/refiner-cache:/src/refiner-cache",
            "r8.im/stability-ai/sdxl@sha256:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        ],
    )
    time.sleep(30)  # Let the docker container startup

num_batches = args.dataset_size // 4
test_batch_idx = int(num_batches * args.dataset_split)
for i in range(num_batches):
    if i < test_batch_idx:
        _dir = train_dir
    else:
        _dir = test_dir
    _prompt = " ".join(random.sample(prompts, 2))
    response = requests.post(
        "http://localhost:5000/predictions",
        headers={"Content-Type": "application/json"},
        json={
            "input": {
                "width": 768,
                "height": 768,
                "prompt": _prompt,
                "refine": "expert_ensemble_refiner",
                "scheduler": "K_EULER",
                "lora_scale": 0.6,
                "num_outputs": 4,
                "guidance_scale": 7.5,
                "apply_watermark": False,
                "high_noise_frac": 0.8,
                "negative_prompt": "drawing, art, illustration",
                "prompt_strength": 1.0,  # 0.8,
                "num_inference_steps": 8,  # 25,
                "disable_safety_checker": True,
            }
        },
    )
    for k in range(4): # Generates 4 images at a time
        img_id = str(uuid.uuid4())[:8]
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"][k].split(",")[1]))
        )
        img = img.resize((336, 336))
        img.save(os.path.join(_dir, f"{img_id}.png"))
        print(f"Saved image {img_id}.png")
        with open(os.path.join(_dir, f"{img_id}.txt"), "w") as f:
            f.write(_prompt)
    print(f"Batch {i+1}/{num_batches} done.")
if sdxl_docker_proc is not None:
    sdxl_docker_proc.terminate()
    os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
