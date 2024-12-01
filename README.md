# Overview

This project 1) evaluates advanced reasoning methods, [Graph of Thought (GoT)](https://github.com/spcl/graph-of-thoughts) and [ReWOO](https://arxiv.org/abs/2305.18323) compared to the naive baselines and 2) conduct error-analysis to improve the advanced reasoning method for the real-world task, on the [Travel Planner](https://github.com/OSU-NLP-Group/TravelPlanner?tab=readme-ov-file) benchmark, which requires generating travel itineraries that meet complex user constraints like budgets and logical sequencing. By comparing these methods to traditional techniques like Chain of Thought (CoT) and Tree of Thought (ToT), we analyze their effectiveness in real-world reasoning tasks and suggest further room to make LLMs handle sophisicated reasoning task. 

# Setup Environment

### 1. Install Necessary Libraries
Run the following command to install all required libraries (conda virtual environment recommended):

```bash
pip3 install openai tqdm geopy langchain pandas torch datasets requests graph_of_thoughts
```


### 2. Download the Database
1. Download the database from [this link](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view).
2. Extract the contents into the `TravelPlanner` directory.

   ```
   YourPathToTravelPlanner
   ```


### 3. Set OpenAI API Key
Export your OpenAI API key as an environment variable. Replace `"Your API Key"` with your actual API key:

```bash
export OPENAI_API_KEY="Your API Key"
```


# How to Run the Code

### 1. Run the Script

Execute the `main.py` file using desired options:

```bash
python3 main.py --llm <LLM_MODEL_NAME> --strategy <STRATEGY> --batch_size <BATCH_SIZE> --output_dir <OUTPUT_DIRECTORY> --is_debug <DEBUG_MODE>
```

#### Options:
- `--llm`: The language model to use (default: `"gpt-4o-mini"`).
- `--strategy`: Reasoning strategy to apply. Options include:
  - `vanilla`: Basic LLM reasoning.
  - `few_shot_llm`: Few-shot prompting with training examples.
  - `got`: Graph of Thought reasoning.
  - `rewoo`: ReWOO reasoning.
- `--batch_size`: Number of queries processed in one batch (default: `2`).
- `--output_dir`: Directory to save results (default: `./res`).
- `--is_debug`: Debug mode. Set to `True` for a quick test run or `False` for full evaluation (default: `True`).

### 2. Example Commands

1. Run with Default Settings:
   ```bash
   python3 main.py
   ```

2. Run with Few-Shot Reasoning:
   ```bash
   python main.py --strategy few_shot_llm --batch_size 4 --output_dir ./output
   ```

3. Run with ReWOO and Full Debug Off:
   ```bash
   python main.py --strategy rewoo --is_debug False
   ```


### 3. Outputs
- Predictions: Saved to `<OUTPUT_DIRECTORY>/generated_predictions-<STRATEGY>.json`.
- Postprocessed Plans: Saved to `<OUTPUT_DIRECTORY>/generated_predictions-<STRATEGY>-postprocess.json`.
- Evaluation Results: Saved to `<OUTPUT_DIRECTORY>/generated_predictions-<STRATEGY>-result.json`.

### 4. Results
TODO

# Contact
If you have any problems, please contact [Wonjoon Choi](mailto:wc2852@columbia.edu) and [Wookje Han](mailto:wh2571@columbia.edu).



