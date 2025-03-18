from main import record
from openai import OpenAI
from argparse import ArgumentParser
from dotenv import load_dotenv
from os import getenv
from termcolor import colored

if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = getenv("OPENAI_API_KEY")
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    messages = []
    while True:
        transcript = record(verbose=args.verbose, cutoff_on_silence=True)
        print(f"[ YOU ] {transcript}")
        messages.append({"role": "user", "content": transcript})
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            #temperature=self.temperature,
            messages=messages,
        )
        response_content = response.choices[0].message.content
        messages.append({
          "role": "assistant",
          "content": response_content
        })
        print()
        print(f"[ANTON] {response_content}")
        _ = input(colored("Press any key to continue", "red", attrs=["bold"]))
