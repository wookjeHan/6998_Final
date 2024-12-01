import os
import re
import logging
import datetime
import json
import csv
from statistics import fmean
from typing import Dict, List, Callable, Set, Union
from .graph_of_thoughts import controller, language_models, operations, prompter, parser
from data import get_dataloader, get_dataset

    
class PlanningPrompter(prompter.Prompter):
    """
    PlanningPrompter provides the generation of prompts specific to the Travel Planning example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    planning_prompt_start = """Give me the Travel Plan based on the following constraints and information. Output only the created Plan between the tags <Plan> and </Plan>, without any additional text.
    
    Here are information:
    """
    planning_prompt_block = """
    {information}
    """

    improve_prompt_start = """The following Plan <S> sets us Travel Plan given constraints and reference information.
    Please improve the Plan <S> by maximizing their accuracy under constraints and information, while minimizing errors. Output only the improved Plan, placed between the two tags <Plan> and </Plan>, without any additional text.
    
    Here are information:
    """

    improve_prompt_block = """
    {information}
    """

    improve_prompt_end = """
    Here is the Plan <S>:
    <S>
    {plan}
    </S>
    """

    score_prompt_base = """The following Plan <S> sets us Travel Plan given constraints and information.
    Please score the Plan <S> in terms of accuracy and efficiency. 
    A score of 10 implies that planning follows all the constraints and information, with perfect efficiency in traveling.
    You may provide reasoning for your scoring, but the final score should be between the tags <Score> and </Score>.
    
    Here are original information:
    """

    score_prompt_block = """
    {information}
    """

    score_prompt_end = """
    Here is the Plan <S>:
    <S>
    {plan}
    </S>
    """

    aggregate_full_prompt_base = """The following Plans <S1> - <S{num_plans}> each sets us Travel Plan given constraints and information.
    Combine the Plans <S1> - <S{num_plans}> into a new one, maximizing their accuracy under constraints and information, while minimizing errors.
    Output only the new Plan between the tags <Plan> and </Plan>, without any additional text.   
    
    Here are the original information:
    """

    aggregate_full_prompt_block1 = """
    {information}
    """
    aggregate_full_prompt_mid = """
    Here are the Plans <S1> - <S{num_plans}>:
    """

    aggregate_full_prompt_block2 = """
    <S{num}>
    {plan}
    </S{num}>
    """

    aggregate_sub_prompt_base = """The following Plans <S1> - <S{num_plans}> are summaries of some other Plans.
    Combine them into a new one, make sure to maximize their accuracy under constraints and information, while minimizing errors.
    Output only the new Plan between the tags <Plan> and </Plan>, without any additional text.
    
    Here are Plans <S1> - <S{num_plans}>:
    """

    aggregate_sub_prompt_generate = """
    Plan <S{num}>:
    {plan}
    </S{num}>
    """

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        """

        if len(state_dicts[0]["parts"]) > 0 and len(state_dicts[0]["parts"]) < len(
            state_dicts[0]["information"]
        ):
            prompt = self.aggregate_sub_prompt_base(
                num_plans=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_sub_prompt_generate.format(
                    plan=state_dict["current"], num=i + 1
                )
            return prompt
        else:
            prompt = self.aggregate_full_prompt_base.format(
                num_plans=len(state_dicts),
            )
            prompt += self.aggregate_full_prompt_block1.format(
                information=state_dicts[0]["information"] # ADD
            )
            prompt += self.aggregate_full_prompt_mid.format(
                num_plans=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_full_prompt_block2.format(
                    plan=state_dict["current"], num=i + 1
                )
            return prompt

    def generate_prompt(
        self,
        num_branches: int,
        information: str,
        method: str,
        parts: Set[str],
        current: str,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param documents: The list of documents to be merged.
        :type documents: List[str]
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param parts: Indices of the already processed document parts.
        :type parts: Set[str]
        :param current: The intermediate solution.
        :type current: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If method is not implemented yet.
        """

        prompt = ""

        if method.startswith("got"):
            if current is None or current == "":
                prompt += self.planning_prompt_start
                prompt += self.planning_prompt_block.format(
                    information=information # ADD
                )
                return prompt
            else:
                prompt += self.improve_prompt_start
                prompt += self.improve_prompt_block.format(
                    information=information # ADD
                )
                prompt += self.improve_prompt_end.format(plan=current)
                return prompt
        else:
            assert False, "Not implemented yet."

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        :raise AssertionError: If more than one thought state is supplied.
        """

        if len(state_dicts) > 1:
            assert False, "Not implemented yet."
        else:
            prompt = self.score_prompt_base
            prompt += self.score_prompt_block.format(information=state_dicts[0]["information"]) # 추가
            prompt += self.score_prompt_end.format(
                plan=state_dicts[0]["current"],
            )
            return prompt

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass


class PlanningParser(parser.Parser):
    """
    PlanningParser provides the parsing of language model reponses specific to the Travel Planning Task.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def strip_answer_helper(self, text: str, tag: str = "") -> str:
        """
        Helper function to remove tags from a text.

        :param text: The input text.
        :type text: str
        :param tag: The tag to be stripped. Defaults to "".
        :type tag: str
        :return: The stripped text.
        :rtype: str
        """

        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        if tag != "":
            start = text.rfind(f"<{tag}>")
            end = text.rfind(f"</{tag}>")
            if start != -1 and end != -1:
                text = text[start + len(f"<{tag}>") : end].strip()
            elif start != -1:
                logging.warning(
                    f"Only found the start tag <{tag}> in answer: {text}. Returning everything after the tag."
                )
                text = text[start + len(f"<{tag}>") :].strip()
            elif end != -1:
                logging.warning(
                    f"Only found the end tag </{tag}> in answer: {text}. Returning everything before the tag."
                )
                text = text[:end].strip()
            else:
                logging.warning(
                    f"Could not find any tag {tag} in answer: {text}. Returning the full answer."
                )
        return text

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        """

        new_states = []
        for text in texts:
            if len(states[0]["parts"]) < len(states[0]["information"]):
                # subpart aggregation
                text = self.strip_answer_helper(text, "Plan")
                new_state = states[0].copy()
                new_state["current"] = text
                new_state["parts"] = set()
                for state in states:
                    new_state["parts"] = new_state["parts"] | state["parts"]

                new_states.append(new_state)
            else:
                # full NDA aggregation
                text = self.strip_answer_helper(text, "Plan")
                new_state = states[0].copy()
                new_state["current"] = text
                new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            text = self.strip_answer_helper(text, "Plan")
            new_state = state.copy()
            new_state["current"] = text
            new_states.append(new_state)
        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        :raise AssertionError: If the number of thought states is not one.
        """
        assert len(states) == 1, "Only one state is allowed for scoring."
        if len(states) == 1:
            # individual scoring
            scores = []
            for text in texts:
                answer = self.strip_answer_helper(text, "Score")
                res = re.findall(r"\d+\.?\d*", answer)
                
                if len(res) == 1:
                    scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        f"Found multiple scores in answer: {text}. Returning the last one."
                    )
                    scores.append(float(res[-1]))
                else:
                    logging.warning(
                        f"Could not find any score in answer: {text}. Ignoring this answer."
                    )
                
            if len(scores) == 0:
                logging.warning(
                    f"Could not find any valid score in any answer. Returning 0.0."
                )
                return [0.0]
            mean_score= fmean(scores)
            return [mean_score]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

def processed_data(data):
    return str({key: data[key] for key in data.keys()})

def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 3))
    operations_graph.append_operation(operations.Score(2, False))
    keep_best = operations.KeepBestN(2, True)
    operations_graph.append_operation(keep_best)
    
    operations_graph.append_operation(operations.Aggregate(2))
    operations_graph.append_operation(operations.Score(2, False))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)

    return operations_graph


def generate(
    method: Callable[[], operations.GraphOfOperations],
    batch
) -> float:
    """
    Controller function that executes GoT method for given Batch
    """

    rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

    for data in rows:
        data = processed_data(data)
        logging.info(f"Running method {method.__name__}")
        lm = language_models.ChatGPT(
            "./graph_of_thoughts/language_models/config.json",
            model_name="chatgpt",
            cache=True,
        )
        operations_graph = method()
        executor = controller.Controller(
            lm,
            operations_graph,
            PlanningPrompter(),
            PlanningParser(),
            {
                "information": data,
                "parts": set(),
                "current": "",
                "method": method.__name__,
            },
        )
        try:
            executor.run()
        except Exception as e:
            logging.error(f"Exception: {e}")


        path = "./res/got_test.json"
        for operation in operations_graph.operations:
            for thought in operation.thoughts:
                thought.state["parts"] = list(thought.state["parts"])
        executor.output_graph(path)

class GoT_LLM:
    def __init__(self, model_name: str):
        """
        Initializes the GoT_LLM with the specified LLM configuration.

        :param llm_config_path: Path to the language model configuration JSON file.
        """
        self.llm = language_models.ChatGPT(
            model_name=model_name,
            cache=True,
        )
        self.prompter = PlanningPrompter()
        self.parser = PlanningParser()
        self.method = got  # Default method

    def generate(self, batch):
        """
        Generate travel plans for the given batch using the GoT method.

        :param batch: A batch of input data.
        :return: List of generated plans.
        """
        rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        generated_plans = []

        for data in rows:
            processed_data = str({key: data[key] for key in data.keys()})
            logging.info(f"Generating plan for data: {processed_data}")

            operations_graph = self.method()
            executor = controller.Controller(
                self.llm,
                operations_graph,
                self.prompter,
                self.parser,
                {
                    "information": processed_data,
                    "parts": set(),
                    "current": "",
                    "method": self.method.__name__,
                },
            )

            try:
                executor.run()
                # Extract the finalized plan from the last operation's thoughts
                final_plan = operations_graph.operations[-1].thoughts[0].state.get("current", "")
                generated_plans.append(final_plan)
            except Exception as e:
                logging.error(f"Error during plan generation: {e}")
                generated_plans.append("")  # Append empty plan in case of an error

        return generated_plans
