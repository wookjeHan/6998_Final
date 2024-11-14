from agents import Vanilla_LLM, FewShot_LLM
from data import get_dataloader, get_dataset
import argparse
from tqdm import tqdm
import os
import json

def main(args):
    train_dataset   = get_dataset("train")
    val_dataloader  = get_dataloader("validation", args.batch_size, True)

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
        model = Vanilla_LLM(args.llm)
    else:
        assert False, f"Strategy: {args.strategy} is not supported"
    # Eval
    preds = []
    for batch in tqdm(val_dataloader):
        pred = model.generate(batch)
        preds.extend(pred)
        break

    # ars.output_dir 에 파일저장
    os.makedirs(args.output_dir, exist_ok=True) 
    output_file = os.path.join(args.output_dir, f'generated_predictions-{args.strategy}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=4, ensure_ascii=False)
    print(f"Predictions saved to {output_file}")


if __name__ == '__main__':
    # TODO1: argparse
    #   model을 뭘로할지도 정해야돼 vanilla, few shot, rewoo, got
    #   OUTPUT PATH 지정
    # TODO2: few shot 모델
    # TODO3: 이거는 할 수 있을지 모르겠어 (eval까지 한번 해보기?)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--strategy", type=str, default="few_shot_llm")

    parser.add_argument("--batch_size", type=int, default=4)


    parser.add_argument("--output_dir", type=str, default="./res")
    
    args = parser.parse_args()
    main(args)