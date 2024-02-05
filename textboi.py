import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--llm", type=str, default="gpt")
args = parser.parse_args()


def import_gpt() -> callable:
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
    return llm    

def import_rep() -> callable:
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
    return llm

if __name__ == "__main__":
    llm = import_gpt() if args.llm == "gpt" else import_rep()
    reply = llm("system", "prompt", 1.2, 64)