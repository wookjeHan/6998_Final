from agents import Vanilla_LLM, FewShot_LLM, ReWoo
from data import get_dataloader, get_dataset
from postprocess import postprocess_plan
from evaluation import eval_fn

import argparse
from tqdm import tqdm
import os
import json

def main(args):
    train_dataset   = get_dataset("train")
    val_dataloader  = get_dataloader("validation", args.batch_size, False)

    if args.strategy == 'vanilla':
        model = Vanilla_LLM(args.llm)
    elif args.strategy == 'few_shot_llm':
        model = FewShot_LLM(args.llm, few_shot_num=3, train_datas=train_dataset)
    # 보류
    # elif args.strategy == 'cot':
    #     model = Vanilla_LLM(args.llm)
    elif args.strategy == 'got':
        model = Vanilla_LLM(args.llm)
    elif args.strategy == 'rewoo':
        model = ReWoo(planner_model=args.llm, solver_model=args.llm)
    else:
        assert False, f"Strategy: {args.strategy} is not supported"
    # Eval
    preds = []
    for id, batch in tqdm(enumerate(val_dataloader)):
        pred = model.generate(batch) # array of string
        dict = [{'idx': id*args.batch_size+x, 'query':batch['query'][x], 'plan':pred[x]} for x in range(len(batch['query']))]
        preds.extend(dict)
        if args.is_debug:
            break
    # ars.output_dir 에 파일저장
    os.makedirs(args.output_dir, exist_ok=True) 
    output_file = os.path.join(args.output_dir, f'generated_predictions-{args.strategy}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=4, ensure_ascii=False)
    print(f"Predictions saved to {output_file}")
    # 저장한 것 post processing하기
    print("Start PostProcessing")
    postprocess_dir = os.path.join(args.output_dir, f'generated_predictions-{args.strategy}-postprocess.json')
    postprocess_plan(output_file, postprocess_dir)
    print(f"Postprocess saved to {postprocess_dir}")
    
    # Start eval
    print("Start Evaluating")
    eval_dir = os.path.join(args.output_dir, f'generated_predictions-{args.strategy}-result.json')
    eval_fn("validation", postprocess_dir, eval_dir, args.is_debug)
    print(f"Eval Result saved to {eval_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--strategy", type=str, default="rewoo")
    parser.add_argument("--is_debug", type=bool, default=True)

    parser.add_argument("--batch_size", type=int, default=2)


    parser.add_argument("--output_dir", type=str, default="./res")
    
    args = parser.parse_args()
    main(args)