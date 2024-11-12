from agents import Vanilla_LLM
from data import get_dataloader

from tqdm import tqdm
def main():
    train_datloader = get_dataloader("train", 16, True)
    val_dataloader = get_dataloader("validation", 16, False)
    preds = []
    model = Vanilla_LLM("gpt-4o-mini")
    # Eval
    for batch in tqdm(val_dataloader):
        pred = model.generate(batch)
        preds.extend(pred)
    print(preds)    
    print(len(preds)) 
       
if __name__ == '__main__':
    # TODO1: argparse
    #   model을 뭘로할지도 정해야돼 vanilla, few shot, rewoo, got
    #   OUTPUT PATH 지정
    # TODO2: few shot 모델
    # TODO3: 이거는 할 수 있을지 모르겠어 (eval까지 한번 해보기?)
    main()