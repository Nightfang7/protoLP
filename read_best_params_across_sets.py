#!/usr/bin/env python3
import pickle
import json

PATH = 'lizz_evaluation_results/best_params_across_sets.pkl'

def main():
    with open(PATH, 'rb') as f:
        data = pickle.load(f)
    summary = {}
    for k, v in data.items():
        avg = v.get('avg', {})
        summary[k] = {
            'accuracy': avg.get('accuracy'),
            'f1': avg.get('f1'),
            'precision': avg.get('precision'),
            'recall': avg.get('recall'),
            'pr_auc': avg.get('pr_auc'),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
