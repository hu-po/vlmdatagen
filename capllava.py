import argparse
import base64
import uuid
import os
import subprocess
import time
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--base_dir", type=str, default="/home/oop/dev/data")
parser.add_argument("--llm", type=str, default="gpt")
args = parser.parse_args()

if args.llm == "gpt":
    from textboi import import_gpt
    llm: callable = import_gpt()
elif args.llm == "rep":
    from textboi import import_rep
    llm: callable = import_rep()
else:
    raise ValueError(f"Unknown llm {args.llm}")
if args.data_dir:
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
docker_ps_process = subprocess.Popen(["docker", "ps"], stdout=subprocess.PIPE)
docker_ps_output, _ = docker_ps_process.communicate()    
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
        with open(os.path.join(_dir, image_path), "rb") as f:
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