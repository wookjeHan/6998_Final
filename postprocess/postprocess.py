import os
import openai
from openai import OpenAI
import math
import sys
import time
from tqdm import tqdm
from typing import Iterable, List, TypeVar
import json
from datasets import load_dataset
from data import get_dataset

openai.api_key = os.getenv("OPENAI_API_KEY")


def process_with_gpt4o_mini(text: str, prefix: str) -> str:
    client = OpenAI()
    """Send the text to GPT-4 (gpt4o-mini) for processing with the given prefix prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prefix},
                {"role": "user", "content": text}
            ],
            max_tokens=1500,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        print(f"Error with OpenAI API: {e}")
        time.sleep(5)  # Wait and try again in case of rate limiting
        return process_with_gpt4o_mini(text, prefix)  # Retry on failure



def postprocess_plan(plan_file, post_file):
    prefix = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
-----EXAMPLE-----
 [{{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }}]
-----EXAMPLE END-----
"""

    processed = []
    generated_plan = json.load(open(plan_file, 'r', encoding='utf-8'))

    for plan in tqdm(generated_plan, desc="Processing plans with GPT-4"):
        # Convert plan to a suitable string format for processing (if necessary)
        plan_text = json.dumps(plan['plan'], ensure_ascii=False)
        # Use GPT-4 for processing
        processed_plan = process_with_gpt4o_mini(plan_text, prefix)
        
        # Parse the processed output as JSON and append to results
        try:
            first_index = processed_plan.find('[')
            last_index = processed_plan.rfind(']')
            processed_plan_json = json.loads(processed_plan[first_index:last_index+1])
            plan['plan'] = processed_plan_json
            processed.append(plan)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from GPT-4 response: {processed_plan}")
            processed.append({"error": "Failed to parse JSON", "original": plan})

    # Save processed results
    with open(post_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)
    print(f"Processed Predictions saved")
