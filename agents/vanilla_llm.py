from openai import OpenAI
import openai
from tqdm import tqdm
import json

import os
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

class Vanilla_LLM:
    def __init__(self, model_name, max_tokens=512, temp=0.1, prompt=None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temp = temp
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = "You are a helpful assistant that plan the travel satisfying the constraint.\nPlease refer to the given information when you plan."
        self.client = OpenAI()

    def generate(self, data_batch):
        responses = []
        for query, reference_information in tqdm(zip(data_batch['query'], data_batch['reference_information'])):
            # Giving Reference Information
            query = f"Query: {query}\n\nReference Information:\n"
            # Postprocessing reference information so that json.loads done properly
            reference_information =  re.sub(r"(\{|, )'([^']+)'(?=:)", r'\1"\2"', reference_information)  # Replace keys
            reference_information = re.sub(r": '([^']+)'", r': "\1"', reference_information)         # Replace values
            reference_information = json.loads(reference_information)
            for ref in reference_information:
                query += f"Title: {ref['Description']}\nContent: {ref['Content']}\n\n"
                
            input_messages = [{"role": "system", "content" : self.prompt}, {"role": "user", "content": query}]
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