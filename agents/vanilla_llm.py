from openai import OpenAI
import openai
from tqdm import tqdm

import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class Vanilla_LLM:
    def __init__(self, model_name, max_tokens=512, temp=0.1, prompt=None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temp = temp
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = "You are a helpful assistant that plan the travel satisfying the constraint."
        self.client = OpenAI()

    def generate(self, data_batch):
        responses = []
        for data in tqdm(data_batch['query']):
            input_messages = [{"role": "system", "content" : self.prompt}, {"role": "user", "content": data}]
            # API CALL -> Generate
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=input_messages,
                max_tokens=self.max_tokens,
                temperature=self.temp
            )
            # Extract the generated solution from the response
            # Return the solution
            responses.append(response.choices[0].message.content)
        return responses