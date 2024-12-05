from evaluation_ablation import eval_fn

import argparse
from tqdm import tqdm
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
def main(args):
    postprocess_dir = os.path.join(args.dir_path, f'generated_predictions-{args.strategy}-postprocess.json')
    
    # Start eval
    print("Start Evaluating")
    eval_dir = os.path.join(args.dir_path, f'generated_predictions-{args.strategy}-result.json')
    commonsense, hard = eval_fn("validation", postprocess_dir, eval_dir, args.is_debug)
    print(f"Eval Result saved to {eval_dir}")
    
    total_true = {k: 0 for k in commonsense['easy'][3].keys()}
    total_false = {k: 0 for k in commonsense['easy'][3].keys()}
    total_common = {k: 0 for k in commonsense['easy'][3].keys()}
    for level in commonsense.keys():
        for day in commonsense[level].keys():
            for k, v in commonsense[level][day].items():
                total_true[k] += v['true']
                total_false[k] += v['false']
                total_common[k] += v['total']
                assert v['true'] + v['false'] == v['total'], "Something Wrong"
    # Draw Fig 1
    keys = list(total_true.keys())
    values = [total_true[k] / total_common[k] * 100 for k in keys]
    plt.plot([k[3:] for k in keys], values, marker='o', linestyle='-', color='b')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.ylabel("Correctness (%)")
    plt.legend()
    plt.savefig("commonsense.png")
    
    total_hard_true = defaultdict(int)
    total_hard_false = defaultdict(int)
    total_hard_common = defaultdict(int)
    # total_har{k: 0 for k in hard['easy'][3].keys()}
    for level in hard.keys():
        for day in hard[level].keys():
            for k, v in hard[level][day].items():
                if 'total' in v.keys():
                    total_hard_true[k] += v['true']
                    total_hard_false[k] += v['false']
                    total_hard_common[k] += v['total']
    plt.clf()
    keys = list(total_hard_true.keys())
    values_hard = [total_hard_true[k] / 420 * 100 for k in keys]
    plt.plot(keys, values_hard, marker='o', linestyle='-', color='b')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.ylabel("Correctness (%)")
    plt.legend()
    plt.savefig("hard.png")
    plt.clf()
    
    
    # We have to compare if two dirs are given
    if args.is_comp:
        postprocess_dir_2 = os.path.join(args.dir_path2, f'generated_predictions-{args.strategy}-postprocess.json')
    
        # Start eval
        print("Start Evaluating")
        eval_dir_2 = os.path.join(args.dir_path2, f'generated_predictions-{args.strategy}-result.json')
        commonsense, hard = eval_fn("validation", postprocess_dir_2, eval_dir_2, args.is_debug)
        print(f"Eval Result saved to {eval_dir_2}")
        
        total_true_2 = {k: 0 for k in commonsense['easy'][3].keys()}
        total_false_2 = {k: 0 for k in commonsense['easy'][3].keys()}
        total_common_2 = {k: 0 for k in commonsense['easy'][3].keys()}
        for level in commonsense.keys():
            for day in commonsense[level].keys():
                for k, v in commonsense[level][day].items():
                    total_true_2[k] += v['true']
                    total_false_2[k] += v['false']
                    total_common_2[k] += v['total']
                    assert v['true'] + v['false'] == v['total'], "Something Wrong"
        # Draw Fig 1
        keys = list(total_true_2.keys())
        values = [total_true[k] / total_common[k] * 100 for k in keys]
        values2 = [total_true_2[k] / total_common_2[k] * 100 for k in keys]
        
        plt.plot([k[3:] for k in keys], values, marker='o', linestyle='-', color='b', label="Before")
        plt.plot([k[3:] for k in keys], values2, marker='o', linestyle='-', color='r', label="Advanced")
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.ylabel("Correctness (%)")
        plt.legend()
        plt.savefig("commonsense_comparison.png")
        
        total_hard_true2 = defaultdict(int)
        total_hard_false2 = defaultdict(int)
        total_hard_common2 = defaultdict(int)
        # total_har{k: 0 for k in hard['easy'][3].keys()}
        for level in hard.keys():
            for day in hard[level].keys():
                for k, v in hard[level][day].items():
                    if 'total' in v.keys():
                        total_hard_true2[k] += v['true']
                        total_hard_false2[k] += v['false']
                        total_hard_common2[k] += v['total']
        plt.clf()
        keys = list(total_hard_true.keys())
        values = [total_hard_true[k] / 420  for k in keys]
        values2 = [total_hard_true2[k] / 420 for k in keys]
        
        plt.plot(keys, values, marker='o', linestyle='-', color='b', label='Before')
        plt.plot(keys, values2, marker='o', linestyle='-', color='r', label='Advanced')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.ylabel("Correctness (%)")
        plt.legend()
        plt.savefig("hard_comparison.png")
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="got")
    parser.add_argument("--is_debug", type=bool, default=False)
    parser.add_argument("--dir_path", type=str, default="./res_before_upgrade")
    parser.add_argument("--dir_path2", type=str, default="./res")
    parser.add_argument("--is_comp", type=bool, default=False)

    args = parser.parse_args()
    main(args)