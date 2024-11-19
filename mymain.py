from agents import GoT_LLM
from data import get_dataloader
import os
import json
import argparse
from tqdm import tqdm

def main(args):
    val_dataloader = get_dataloader("validation", args.batch_size, False)

    if args.strategy == 'got':
        model = GoT_LLM(args.llm)

    # pred
    preds = []
    for id, batch in tqdm(enumerate(val_dataloader)):
        pred = model.generate(batch)  # Generate plans
        dicts = [{'idx': id * args.batch_size + x, 'query': batch['query'][x], 'plan': pred[x]} for x in range(len(batch['query']))]
        preds.extend(dicts)
        if args.is_debug:
            break

    # Output results
    if args.output_dir:
        output_path = os.path.join(args.output_dir, "results.json")
        with open(output_path, "w") as f:
            json.dump(preds, f, indent=4)
        print(f"Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="./graph_of_thoughts/language_models/config.json")
    parser.add_argument("--strategy", type=str, default="got")
    parser.add_argument("--is_debug", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./res")

    args = parser.parse_args()
    main(args)
