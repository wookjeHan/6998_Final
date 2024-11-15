from openai import OpenAI
import openai
from tqdm import tqdm

import os
import random

openai.api_key = os.getenv("OPENAI_API_KEY")

class FewShot_LLM:
    def __init__(self, model_name, max_tokens=512, temp=0.1, prompt=None, few_shot_num=1, train_datas=None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temp = temp
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = "You are a helpful assistant that plan the travel satisfying the constraint. When answering user questions follow these examples:"
        self.client = OpenAI()
        
        assert type(few_shot_num) == int and few_shot_num > 0, "Please make few_shot_num bigger than 1. If you want to set as 0, consider using Vanilla LLM"
        assert train_datas is not None, "Should give train data where we pick shots"

        data_size = len(train_datas)
        assert data_size >= few_shot_num, f"Few shot num {few_shot_num} cannot exceed data size {data_size}"
        
        few_shot_indexes = random.sample(range(data_size), few_shot_num)
        
        # Appending few shots
        for ind in few_shot_indexes:
            self.prompt += f"\nQuery: {train_datas['query'][ind]}\nAnswer:{train_datas['annotated_plan'][ind]}\n\n"
                    
    def generate(self, data_batch):
        responses = []
        for data in tqdm(data_batch['query']):
            input_messages = [{"role": "system", "content" : self.prompt}]
            # Appending few shots 
            input_messages.append({"role": "user", "content": data})

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