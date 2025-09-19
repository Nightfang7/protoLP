#!/usr/bin/env python3
"""
Unified runner for Lizz_data:
- Evaluate across datasets/ways/shots with a single set of params
- Optional grid search over parameter lists

It uses evaluate_lizz_35_cases_tuned from test_lizz_data_evaluation_unbalanced_tuned.py
"""

import os
import json
import pickle
import argparse
from itertools import product
from datetime import datetime

from test_lizz_data_evaluation_unbalanced_tuned import evaluate_lizz_35_cases_tuned

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_PARAMS = dict(
    beta=0.5,
    reduction_dim=40,
    lam=10,
    alpha=0.2,
    n_epochs=20,
    use_power_transform=True,
    use_unitary_scaling=True,
    use_centering=True,
    use_map_optimization=False,
    distance_metric='euclidean',
)

def parse_args():
    p = argparse.ArgumentParser(description="Unified runner for Lizz_data evaluation/tuning")
    p.add_argument('--data_root', default='Lizz_data')
    p.add_argument('--autoencoder_dir', default='autoencoder')
    p.add_argument('--output_dir', default='lizz_evaluation_results')
    p.add_argument('--seed', type=int, default=42)

    # datasets/ways/shots
    p.add_argument('--datasets', nargs='*', default=['00177', 'apidms'])
    p.add_argument('--shots', nargs='*', default=['5shots', '10shots'])
    p.add_argument('--ways', nargs='*', type=int, default=[5, 10])

    # mode
    p.add_argument('--grid', action='store_true', help='Enable grid search')

    # single params (for eval mode)
    p.add_argument('--beta', type=float, default=DEFAULT_PARAMS['beta'])
    p.add_argument('--reduction_dim', type=int, default=DEFAULT_PARAMS['reduction_dim'])
    p.add_argument('--lam', type=float, default=DEFAULT_PARAMS['lam'])
    p.add_argument('--alpha', type=float, default=DEFAULT_PARAMS['alpha'])
    p.add_argument('--n_epochs', type=int, default=DEFAULT_PARAMS['n_epochs'])
    p.add_argument('--distance_metric', choices=['euclidean','cosine'], default=DEFAULT_PARAMS['distance_metric'])
    p.add_argument('--no_power', action='store_true')
    p.add_argument('--no_unitary', action='store_true')
    p.add_argument('--no_centering', action='store_true')
    p.add_argument('--use_map', action='store_true')

    # grid lists (used only if --grid)
    p.add_argument('--beta_list', nargs='*', type=float)
    p.add_argument('--reduction_dim_list', nargs='*', type=int)
    p.add_argument('--lam_list', nargs='*', type=float)
    p.add_argument('--alpha_list', nargs='*', type=float)
    p.add_argument('--n_epochs_list', nargs='*', type=int)
    p.add_argument('--distance_list', nargs='*', choices=['euclidean','cosine'])

    return p.parse_args()


def build_params_from_args(args):
    params = dict(
        beta=args.beta,
        reduction_dim=args.reduction_dim,
        lam=args.lam,
        alpha=args.alpha,
        n_epochs=args.n_epochs,
        use_power_transform=not args.no_power,
        use_unitary_scaling=not args.no_unitary,
        use_centering=not args.no_centering,
        use_map_optimization=args.use_map,
        distance_metric=args.distance_metric,
    )
    return params


def run_eval(args):
    params = build_params_from_args(args)
    results = {}

    for dataset_name, shot, way in product(args.datasets, args.shots, args.ways):
        key = f"{dataset_name}_{shot}_{way}way"
        print(f"\n=== Evaluate: {key} ===")
        try:
            case_results, avg_results = evaluate_lizz_35_cases_tuned(
                args.data_root, dataset_name, shot, way,
                args.autoencoder_dir, args.seed, **params
            )
            results[key] = dict(avg=avg_results, cases=case_results, params=params)
            print(f"Avg -> Acc={avg_results['accuracy']:.4f}, F1={avg_results['f1']:.4f}, PR-AUC={avg_results['pr_auc']:.4f}")
        except Exception as e:
            print(f"Failed: {key} -> {e}")
            results[key] = dict(error=str(e), params=params)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pkl = os.path.join(args.output_dir, f'unified_eval_{ts}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    # also print concise json summary
    summary = {k: v['avg'] if 'avg' in v else {'error': v.get('error')} for k, v in results.items()}
    out_json = os.path.join(args.output_dir, f'unified_eval_{ts}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_pkl}\nSaved: {out_json}")


def run_grid(args):
    # fall back to defaults if list not provided
    beta_list = args.beta_list or [0.5]
    rd_list = args.reduction_dim_list or [40]
    lam_list = args.lam_list or [10]
    alpha_list = args.alpha_list or [0.2]
    ne_list = args.n_epochs_list or [20]
    dist_list = args.distance_list or ['euclidean']

    grid = list(product(beta_list, rd_list, lam_list, alpha_list, ne_list, dist_list))
    print(f"Grid size: {len(grid)} param combos")

    all_results = {}
    for dataset_name, shot, way in product(args.datasets, args.shots, args.ways):
        key_ds = f"{dataset_name}_{shot}_{way}way"
        all_results[key_ds] = {}
        print(f"\n=== Grid on: {key_ds} ===")
        for (beta, rd, lam, alpha, ne, dist) in grid:
            params = dict(
                beta=beta,
                reduction_dim=rd,
                lam=lam,
                alpha=alpha,
                n_epochs=ne,
                use_power_transform=not args.no_power,
                use_unitary_scaling=not args.no_unitary,
                use_centering=not args.no_centering,
                use_map_optimization=args.use_map,
                distance_metric=dist,
            )
            name = f"b{beta}_rd{rd}_lam{lam}_a{alpha}_ne{ne}_{dist}"
            print(f"- {name}")
            try:
                _, avg = evaluate_lizz_35_cases_tuned(
                    args.data_root, dataset_name, shot, way,
                    args.autoencoder_dir, args.seed, **params
                )
                all_results[key_ds][name] = dict(params=params, avg=avg)
                print(f"  Acc={avg['accuracy']:.4f}, F1={avg['f1']:.4f}, PR-AUC={avg['pr_auc']:.4f}")
            except Exception as e:
                all_results[key_ds][name] = dict(params=params, error=str(e))
                print(f"  Failed: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pkl = os.path.join(args.output_dir, f'unified_grid_{ts}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    out_json = os.path.join(args.output_dir, f'unified_grid_{ts}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_pkl}\nSaved: {out_json}")


def main():
    args = parse_args()
    if args.grid:
        run_grid(args)
    else:
        run_eval(args)

if __name__ == '__main__':
    main()
