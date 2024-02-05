import argparse
import base64
import uuid
import os
import subprocess
import time
import requests

from io import BytesIO
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--base_dir", type=str, default="/home/oop/dev/data")
parser.add_argument("--num_categories", type=int, default=8)
parser.add_argument("--dataset_size", type=int, default=800)
parser.add_argument("--dataset_split", type=float, default=0.8)
parser.add_argument("--llm", type=str, default="gpt")
args = parser.parse_args()

dataset_id = str(uuid.uuid4())[:6]
print(f"No data directory specified, generating new dataset {dataset_id}")
data_dir = os.path.join(args.base_dir, f"vlmgen.{dataset_id}")
os.makedirs(data_dir, exist_ok=True)
print(f"data directory at {data_dir}")
train_dir = os.path.join(data_dir, "train")
os.makedirs(train_dir, exist_ok=True)
print(f"train directory at {train_dir}")
test_dir = os.path.join(data_dir, "test")
os.makedirs(test_dir, exist_ok=True)
print(f"test directory at {test_dir}")


# -------------- LLM
if args.llm == "gpt":
    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def llm(system: str, prompt: str, temp: float, max_tokens: int):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4-1106-preview",
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

elif args.llm == "codellama":
    # https://replicate.com/meta/codellama-70b-instruct
    import replicate

    def llm(system: str, prompt: str, temp: float, max_tokens: int):
        output = replicate.run(
            "meta/codellama-70b-instruct:a279116fe47a0f65701a8817188601e2fe8f4b9e04a518789655ea7b995851bf",
            input={
                "top_k": 10,
                "top_p": 0.95,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temp,
                "system_prompt": system,
                "repeat_penalty": 1.1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
        )
        return output



# # Use llm to generate categories
# unique_categories = set()
# while len(unique_categories) < args.num_categories:
#     reply = llm(
#         """
# You are a sampling machine that provides perfectly sampled words. 
# You provide samples from the distribution of semantic visual concepts. 
# Reply only with lowercase single words.
#         """,
#         """
# Return a comma separated list of 10 words with no spaces.
# These words will be used as classes for an image classification task. 
#         """,
#         1.2,
#         64,
#     )
#     unique_categories.update(set([_.lower() for _ in reply.split(",")]))

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
for i in range(num_batches):
    if i < num_batches * args.dataset_split:
        _dir = train_dir
    else:
        _dir = test_dir
    response = requests.post(
        "http://localhost:5000/predictions",
        headers={"Content-Type": "application/json"},
        json={
            "input": {
                "width": 768,
                "height": 768,
                "prompt": "webcam image of a human striking a pose",
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
if sdxl_docker_proc is not None:
    sdxl_docker_proc.terminate()
    os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")


# -------------- LLaVA
if "llava" in docker_ps_output.decode():
    print(f"Docker for LLaVA is already running: {docker_ps_output.decode()}")
    llava_docker_proc = None
else:
    os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
    llava_docker_proc = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "-p",
            "5000:5000",
            "--gpus=all",
            "-v",
            "/home/oop/dev/data/llava-v1.6-mistral-7b/:/src/liuhaotian/llava-v1.6-mistral-7b/",
            "-v",
            "/home/oop/dev/data/clip-vit-large-patch14-336:/src/openai/clip-vit-large-patch14-336"
            "r8.im/yorickvp/llava-v1.6-mistral-7b@sha256:4798da673efa7bc088aa046c2d5382d0c8b4fad971c828c3740d44feb7cbb471",
        ],
    )
    time.sleep(30)  # Let the docker container startup
for _dir in (train_dir, test_dir):
    for image_path in os.listdir(_dir):
        if not image_path.endswith(".png"):
            continue
        img_id = image_path.split(".")[0]
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        response = requests.post(
            "http://localhost:5000/predictions",
            headers={"Content-Type": "application/json"},
            json={
                "input": {
                    "image": f"data:image/jpeg;base64,{base64_image}",
                    "top_p": 1,
                    "prompt": "Describe the image",
                    "max_tokens": 1024,
                    "temperature": 0.2,
                }
            },
        )
        output = response.json()["output"][0]
        print(output)
        caption_filepath = os.path.join(_dir, f"{img_id}.txt")
        with open(caption_filepath, "w") as f:
            f.write(output)
if llava_docker_proc is not None:
    llava_docker_proc.terminate()
    os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
