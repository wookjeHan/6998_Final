import os
import time

import openai 
from openai import OpenAI

OPENAI_CONFIG = {
    "temperature": 0.5,
    "max_tokens": 1024,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

openai.api_key = os.getenv("OPENAI_API_KEY")

class Node:
    def __init__(self, name, input_type, output_type):
        self.name = name
        self.input_type = input_type
        self.output_type = output_type

    def run (self, input, log=False):
        raise NotImplementedError


class LLMNode(Node):
    def __init__(self, name="BaseLLMNode", model_name="gpt-4o-mini", stop=None, input_type=str, output_type=str):
        super().__init__(name, input_type, output_type)
        self.model_name = model_name
        self.stop = stop
        self.client = OpenAI()
        
    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        response = self.call_llm(input, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion

    def call_llm(self, prompt, stop):
        messages = [{"role": "user", "content": prompt}]
        response = None
        while response is None:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=OPENAI_CONFIG["temperature"],
                    max_tokens=OPENAI_CONFIG["max_tokens"],
                    top_p=OPENAI_CONFIG["top_p"],
                    frequency_penalty=OPENAI_CONFIG["frequency_penalty"],
                    presence_penalty=OPENAI_CONFIG["presence_penalty"],
                    stop=stop
                )
            except Exception as e:
                print(e)
                print("Retrying")
                time.sleep(10)
        return {"input": prompt,
                "output": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens}
        
