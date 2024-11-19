import time
from .rewoo_utils import fewshots, LLMNode, planner_prompts, solver_prompts, Node
from openai import OpenAI
import requests
import re
from tqdm import tqdm
import json

from geopy.geocoders import Nominatim
from langchain import OpenAI, LLMMathChain, LLMChain, PromptTemplate
from langchain.agents import Tool

class CalculatorWorker(Node):
    def __init__(self, name="Calculator"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.description = "A calculator that can compute arithmetic expressions. Useful when you need to perform " \
                           "math calculations. Input should be a mathematical expression"

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        llm = OpenAI(temperature=0)
        response = llm.invoke(f"Please extract the mathematical form from this query. Your answer should be formatted as mathematical: \nQuery: {input}")
        tool = LLMMathChain(llm=llm, verbose=False)
        try:
            response = tool(response)
            evidence = response["answer"].replace("Answer:", "").strip()
        except Exception as e:
            evidence = "Cannot calculate well"
        assert isinstance(evidence, self.output_type)
        if log:
            return {"input": response["question"], "output": response["answer"]}
        return evidence


class LLMWorker(Node):
    def __init__(self, name="LLM"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.description = "A pretrained LLM like yourself. Useful when you need to act with general world " \
                           "knowledge and common sense. Prioritize it when you are confident in solving the problem " \
                           "yourself. Input can be any instruction."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        llm = OpenAI(temperature=0)
        prompt = PromptTemplate(template="Respond in short directly with no extra words.\n\n{request}",
                                input_variables=["request"])
        tool = LLMChain(prompt=prompt, llm=llm, verbose=False)
        response = tool(input)
        evidence = response["text"].strip("\n")
        assert isinstance(evidence, self.output_type)
        if log:
            return {"input": response["request"], "output": response["text"]}
        return evidence
    
class ReferenceWorker(Node):
    def __init__(self, name="Reference", reference=None):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "This is the model that provides the reference information for planning the travel. It can provide the information such as hotels, flight, attractions, price of restaurant, etc."
        assert reference is not None, "Why reference is None?"
        self.reference = reference
        
    def run(self, input, log=False):
        llm = OpenAI(temperature=0)
        evidence = llm.invoke(f"From the given information please search the query\n\nInformation: {self.reference}\n\nQuery:{input}")
        assert isinstance(evidence, self.output_type)
        if log:
            return {"input": input, "output": evidence}
        return evidence
    
WORKER_REGISTRY = {"Calculator": CalculatorWorker,
                   "LLM": LLMWorker,
                   "Reference": ReferenceWorker
                   }



class Planner(LLMNode):
    def __init__(self, workers, prefix=planner_prompts.DEFAULT_PREFIX, suffix=planner_prompts.DEFAULT_SUFFIX, fewshot=planner_prompts.DEFAULT_FEWSHOT,
                 model_name="gpt-4o-mini", stop=None):
        super().__init__("Planner", model_name, stop, input_type=str, output_type=str)
        self.workers = workers
        self.prefix = prefix
        self.suffix = suffix
        self.fewshot = fewshot

    def run(self, input, reference, log=False):
        self.worker_prompt = self._generate_worker_prompt(reference)
        assert isinstance(input, self.input_type)
        prompt = self.prefix + self.worker_prompt + self.fewshot + self.suffix + input + '\n'
        
        response = self.call_llm(prompt, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion

    def _get_worker(self, name, reference=None):
        if name in WORKER_REGISTRY and reference is not None:
            return WORKER_REGISTRY[name](reference=reference)
        elif name in WORKER_REGISTRY:
            return WORKER_REGISTRY[name]()
        else:
            raise ValueError("Worker not found")

    def _generate_worker_prompt(self, reference):
        prompt = "Tools can be one of the following:\n"
        for name in self.workers:
            if name == "Reference":
                worker = self._get_worker(name, reference)
            else:
                worker = self._get_worker(name)
            prompt += f"{worker.name}[input]: {worker.description}\n"
        return prompt + "\n"


class Solver(LLMNode):
    def __init__(self, prefix=solver_prompts.DEFAULT_PREFIX, suffix=solver_prompts.DEFAULT_SUFFIX, model_name="gpt-4o-mini", stop=None):
        super().__init__("Solver", model_name, stop, input_type=str, output_type=str)
        self.prefix = prefix
        self.suffix = suffix

    def run(self, input, worker_log, log=False):
        assert isinstance(input, self.input_type)
        prompt = self.prefix + input + "\n" + worker_log + self.suffix + input + '\n'
        response = self.call_llm(prompt, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion


class PWS:
    def __init__(self, available_tools=["Reference", "LLM", "Calculator"], fewshot="\n", planner_model="gpt-4o-mini",
                 solver_model="gpt-4o-mini"):
        self.workers = available_tools
        self.planner = Planner(workers=self.workers,
                               model_name=planner_model,
                               fewshot=fewshot)
        self.solver = Solver(model_name=solver_model)
        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}
        self.tool_counter = {}

    # input: the question line. e.g. "Question: What is the capital of France?"
    def run(self, input, reference):
        # run is stateless, so we need to reset the evidences
        self._reinitialize()
        result = {}
        st = time.time()
        # Plan
        planner_response = self.planner.run(input, reference, log=True)
        plan = planner_response["output"]
        planner_log = planner_response["input"] + planner_response["output"]
        self.plans = self._parse_plans(plan)
        self.planner_evidences = self._parse_planner_evidences(plan)
        #assert len(self.plans) == len(self.planner_evidences)
        # Work
        self._get_worker_evidences(reference)
        worker_log = ""
        for i in range(len(self.plans)):
            e = f"#E{i + 1}"
            worker_log += f"{self.plans[i]}\nEvidence:\n{self.worker_evidences[e]}\n"
        # Solve
        solver_response = self.solver.run(input, worker_log, log=True)
        output = solver_response["output"]
        solver_log = solver_response["input"] + solver_response["output"]

        result["wall_time"] = time.time() - st
        result["input"] = input
        result["output"] = output
        result["planner_log"] = planner_log
        result["worker_log"] = worker_log
        result["solver_log"] = solver_log
        result["tool_usage"] = self.tool_counter
        result["steps"] = len(self.plans) + 1
        result["total_tokens"] = planner_response["prompt_tokens"] + planner_response["completion_tokens"] \
                                 + solver_response["prompt_tokens"] + solver_response["completion_tokens"] \
                                 + self.tool_counter.get("LLM_token", 0) \
                                 + self.tool_counter.get("Calculator_token", 0)
                                 
        return result

    def _parse_plans(self, response):
        plans = []
        for line in response.splitlines():
            pattern = r"^Plan\s+\d+:"
            pattern_2 = r"^###\s+Plan\s+\d+"
            if line.startswith("Plan:") or re.match(pattern, line) or re.match(pattern_2, line):
                plans.append(line)
        return plans

    def _parse_planner_evidences(self, response):
        evidences = {}
        for line in response.splitlines():
            if line.startswith("#") and line[1] == "E" and line[2].isdigit():
                e, tool_call = line.split("=", 1)
                e, tool_call = e.strip(), tool_call.strip()
                if len(e) == 3:
                    evidences[e] = tool_call
                else:
                    evidences[e] = "No evidence found"
        return evidences

    # use planner evidences to assign tasks to respective workers.
    def _get_worker_evidences(self, reference):
        for e, tool_call in self.planner_evidences.items():
            if "[" not in tool_call:
                self.worker_evidences[e] = tool_call
                continue
            tool, tool_input = tool_call.split("[", 1)
            tool_input = tool_input[:-1]
            # find variables in input and replace with previous evidences
            for var in re.findall(r"#E\d+", tool_input):
                if var in self.worker_evidences:
                    tool_input = tool_input.replace(var, "[" + self.worker_evidences[var] + "]")
            if tool in self.workers:
                if tool == "Reference":
                    self.worker_evidences[e] = WORKER_REGISTRY[tool](reference=reference).run(tool_input)
                else:
                    self.worker_evidences[e] = WORKER_REGISTRY[tool]().run(tool_input)
            else:
                self.worker_evidences[e] = "No evidence found"

    def _reinitialize(self):
        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}
        self.tool_counter = {}

    def generate(self, data_batch):
        responses = []
        for query, reference_information in tqdm(zip(data_batch['query'], data_batch['reference_information'])):
            reference = ""
            # Postprocess reference information
            reference_information =  re.sub(r"(\{|, )'([^']+)'(?=:)", r'\1"\2"', reference_information)  # Replace keys
            reference_information = re.sub(r": '([^']+)'", r': "\1"', reference_information)         # Replace values
            reference_information = json.loads(reference_information)
            for ref in reference_information:
                reference += f"Title: {ref['Description']}\nContent: {ref['Content']}\n\n"
            res = self.run(query, reference)['output']
            responses.append(res)
            
        return responses

class ReWoo(PWS):
    def __init__(self, fewshot=fewshots.HOTPOTQA_PWS_BASE, planner_model="gpt-4o-mini",
                 solver_model="gpt-4o-mini", available_tools=["Reference", "LLM", "Calculator"]):
        super().__init__(available_tools=available_tools,
                         fewshot=fewshot,
                         planner_model=planner_model,
                         solver_model=solver_model)