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
  - `got_advanced`: Advanced GoT reasoning.
  - `rewoo_advanced`: Advanced ReWOO reasoning.
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
| Method              | Delivery Rate | Commonsense Constraint (Micro) | Commonsense Constraint (Macro) | Hard Constraint (Micro) | Hard Constraint (Macro) | Final Pass |
|---------------------|---------------|---------------------------------|---------------------------------|--------------------------|--------------------------|------------|
| Vanilla LLMs        | 100.00%       | 60.83%                          | 0.56%                           | 0.23%                   | 0.00%                   | 0.00%      |
| Few Shot LLMs       | 100.00%       | 65.83%                          | 2.78%                           | 3.33%                   | 1.67%                   | 0.00%      |
| ReWOO               | 100.00%       | 70.28%                          | 6.67%                           | 5.00%                   | 1.67%                   | 1.11%      |
| GoT                 | 100.00%       | 73.33%                          | 6.67%                           | 5.48%                   | 1.67%                   | 1.11%      |
| Advanced ReWOO      | 100.00%       | 72.50%                          | 6.67%                           | 5.95%                   | 3.89%                   | 1.67%      |
| Advanced GoT        | 100.00%       | **74.58%**                      | **6.67%**                       | **7.38%**               | **4.44%**               | **1.67%**  |



### 5. Ablation

```bash
python3 ablation.py --strategy <STRATEGY> --dir_path <PATH_TO_RESULT_DIR> --dir_path2 <PATH_TO_RESULT_DIR2> --is_comp <COMPARISON MODE>
```


#### Options:
- `--strategy`: Reasoning strategy to apply. Options include:
  - `vanilla`: Basic LLM reasoning.
  - `few_shot_llm`: Few-shot prompting with training examples.
  - `got`: Graph of Thought reasoning.
  - `rewoo`: ReWOO reasoning.
  - `got_advanced`: Advanced GoT reasoning.
  - `rewoo_advanced`: Advanced ReWOO reasoning.
- `--dir_path`: Directory that save results file to analyze 
- `--dir_path2`: Directory that save results file to analyze (needed if it is comparison mode)
- `--is_comp`: Bool value to indicate whether you want to compare the result between dir_path and dir_path2

#### Example Results:

**GoT's Result**
![Comparison between GoT and Advanced within Commonsense](/evaluation_ablation/got/commonsense_comparison.png)
**Figure 1:** Comparison between GoT and Advanced within Commonsense

![Comparison between GoT and Advanced within Hard](/evaluation_ablation/got/hard_comparison.png)
**Figure 2:** Comparison between GoT and Advanced within Hard

ReWOO's Result
![Comparison between ReWOO and Advanced within Commonsense](/evaluation_ablation/rewoo/commonsense_comparison.png)
**Figure 3:** Comparison between ReWOO and Advanced within Commonsense

![Comparison between ReWOO and Advanced within Hard](/evaluation_ablation/rewoo/hard_comparison.png)
**Figure 4:** Comparison between ReWOO and Advanced within Hard



# Contact
If you have any problems, please contact [Wonjoon Choi](mailto:wc2852@columbia.edu) and [Wookje Han](mailto:wh2571@columbia.edu).



