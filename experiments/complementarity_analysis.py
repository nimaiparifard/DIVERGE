# Experiment #3 (R1-C2 / R2-C1 / R2-C2 / R4-C1): is the init diversity *useful*?
# Complementarity of the members and ensemble gain, multi-init vs seed-only ensembles.
"""
Disagreement alone is not enough (a weak model disagrees a lot). Here we check
that the diversity is complementary and turns into ensemble gain.

Uses the shared GNN pool (one GNN per init x seed, fixed GNN seed/split).
All ensembles are UNIFORM averages of the members' class probabilities, so no
ensemble weights are learned and nothing is fitted on test nodes.

  a) Pairwise complementarity on test nodes: double-fault rate (both wrong),
     oracle accuracy (at least one right), oracle gain over the better member,
     Yule's Q statistic (Kuncheva & Whitaker, 2003). Seed-only vs init-only
     pairs, same test as experiment #1.
  b) Seed-only ensembles (one init, k seeds) vs multi-init ensembles (k inits,
     one seed) at the SAME size k = min(#inits, #seeds), plus the full-size
     versions. For each ensemble, the ambiguity decomposition
     (Krogh & Vedelsby, 1995) with squared loss on the probability vectors:
          ensemble error = average member error - ambiguity (diversity)
     which is an exact identity for a uniform average, and links diversity
     to the gain directly.
  c) Diversity vs gain: every k-subset of the pool (sampled if too many),
     mean pairwise disagreement vs gain over the mean member, Spearman
     correlation, and the average gain grouped by number of distinct inits.

Outputs (under --out_dir, default results/diversity/complementarity/):
  complementarity_summary.csv     one row per run
  complementarity_ensembles.csv   one row per seed-only / multi-init ensemble
  complementarity_subsets.csv     diversity-vs-gain summary per (k, #distinct inits)
  <tag>_*_heatmap.png, <tag>_diversity_vs_gain_k<k>.png, <tag>_report.txt
"""
import itertools
import math
import os

import numpy as np
import torch
from scipy.stats import spearmanr, mannwhitneyu

from experiments.diversity_common import (
    add_common_args, pool_from_args, parse_list, within_between_test, save_pool_heatmap, save_matrix_csv,
    run_info, append_rows, TextReport,
)


def pairwise_complementarity(correct, preds):
    """Pairwise matrices on test nodes (rows of `correct`/`preds` = members)."""
    M = correct.shape[0]
    c = correct.astype(np.float64)
    n11 = c @ c.T                       # both right
    n00 = (1 - c) @ (1 - c).T           # both wrong
    n10 = c @ (1 - c).T                 # i right, j wrong
    n01 = n10.T
    n = correct.shape[1]
    denom = n11 * n00 + n01 * n10
    Q = np.where(denom > 0, (n11 * n00 - n01 * n10) / np.where(denom > 0, denom, 1), 1.0)
    acc = c.mean(axis=1)
    oracle = 1 - n00 / n
    disagree = np.stack([(preds[i:i + 1] != preds).mean(axis=1) for i in range(M)])
    return {
        'double_fault': n00 / n,
        'oracle_acc': oracle,
        'oracle_gain': oracle - np.maximum(acc[:, None], acc[None, :]),
        'q_statistic': Q,
        'disagreement': disagree,
    }


def ensemble_stats(probs, y_onehot, y, members):
    """Uniform-average ensemble of `members` (indices into probs [M, n, C]) on the given nodes."""
    P = probs[members]                                   # [k, n, C]
    p_bar = P.mean(dim=0)
    member_acc = (P.argmax(dim=2) == y[None]).float().mean(dim=1)
    member_err = ((P - y_onehot[None]) ** 2).sum(dim=2).mean(dim=1)
    ambiguity = ((P - p_bar[None]) ** 2).sum(dim=2).mean(dim=1)
    ens_acc = (p_bar.argmax(dim=1) == y).float().mean().item()
    return {
        'ens_acc': ens_acc * 100,
        'mean_member_acc': member_acc.mean().item() * 100,
        'best_member_acc': member_acc.max().item() * 100,
        'gain_vs_mean': (ens_acc - member_acc.mean().item()) * 100,
        'gain_vs_best': (ens_acc - member_acc.max().item()) * 100,
        'ens_sq_err': ((p_bar - y_onehot) ** 2).sum(dim=1).mean().item(),
        'avg_member_sq_err': member_err.mean().item(),
        'ambiguity': ambiguity.mean().item(),
    }


def subset_sweep(probs, y, acc, disagree, inits, k, max_subsets, rng, chunk=512):
    """Diversity vs gain over k-subsets of the pool."""
    M = probs.shape[0]
    total = math.comb(M, k)
    if total <= max_subsets:
        subsets = np.array(list(itertools.combinations(range(M), k)))
    else:
        seen = set()
        while len(seen) < max_subsets:
            seen.add(tuple(sorted(rng.choice(M, size=k, replace=False))))
        subsets = np.array(sorted(seen))
    ens_acc = []
    for s in range(0, len(subsets), chunk):
        idx = torch.as_tensor(subsets[s:s + chunk])
        p_bar = probs[idx].mean(dim=1)                   # [B, n, C]
        ens_acc.append((p_bar.argmax(dim=2) == y[None]).float().mean(dim=1).numpy() * 100)
    ens_acc = np.concatenate(ens_acc)
    iu = np.triu_indices(k, 1)
    div = np.array([disagree[np.ix_(s, s)][iu].mean() for s in subsets])
    mean_acc = acc[subsets].mean(axis=1)
    best_acc = acc[subsets].max(axis=1)
    n_inits = np.array([len(set(inits[s])) for s in subsets])
    return {'subsets': subsets, 'diversity': div, 'ens_acc': ens_acc, 'gain_vs_mean': ens_acc - mean_acc,
            'gain_vs_best': ens_acc - best_acc, 'n_inits': n_inits, 'sampled': total > max_subsets, 'total': total}


def plot_div_vs_gain(res, k, save_path, pool):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in sorted(set(res['n_inits'])):
        m = res['n_inits'] == n
        ax.scatter(res['diversity'][m] * 100, res['gain_vs_mean'][m], s=8, alpha=0.5, label=f'{n} distinct init(s)')
    ax.set_xlabel('mean pairwise disagreement (%)')
    ax.set_ylabel('ensemble gain over mean member (pp)')
    ax.set_title(f'Diversity vs ensemble gain, k={k}\n{pool.dataset_name} / {pool.llm_name}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] plot saved to: {save_path}')


def main(args):
    pool = pool_from_args(args)
    out_dir = os.path.join(args.out_dir, 'complementarity')
    report = TextReport(os.path.join(out_dir, f'{pool.tag}_report.txt'),
                        f'Complementarity & ensemble gain | {pool.dataset_name} | {pool.llm_name}')
    for line in pool.describe():
        report.line(line)
    info = run_info(pool)
    rng = np.random.default_rng(0)

    gnn = pool.load_gnn_pool(retrain=args.retrain_gnn)
    test = gnn['test_mask']
    y = gnn['y'][test]
    probs = gnn['probs'][:, test]                        # [M, n_test, C]
    preds = gnn['preds'][:, test].numpy()
    correct = preds == y.numpy()[None]
    acc = correct.mean(axis=1) * 100
    y_onehot = torch.nn.functional.one_hot(y, probs.shape[2]).float()

    # a) pairwise complementarity
    pw = pairwise_complementarity(correct, preds)
    report.section('a) Pairwise complementarity (test nodes)')
    summary = dict(info, n_test_nodes=int(test.sum()))
    # orientation for the test: larger = more diverse / complementary
    oriented = {'double_fault_complement': 1 - pw['double_fault'], 'oracle_gain': pw['oracle_gain'],
                'q_distance': (1 - pw['q_statistic']) / 2}
    for name, D in oriented.items():
        wb = within_between_test(D, pool.inits, pool.seeds, n_perm=args.n_perm, rng=rng)
        report.within_between(name, wb)
        summary.update(TextReport.within_between_row(wb, f'{name}_'))
    for name in ('double_fault', 'oracle_acc', 'q_statistic'):
        save_matrix_csv(pw[name], pool.labels, os.path.join(out_dir, f'{pool.tag}_{name}_matrix.csv'))
        if not args.no_plots:
            save_pool_heatmap(pw[name], pool.labels, pool.inits,
                              os.path.join(out_dir, f'{pool.tag}_{name}_heatmap.png'),
                              title=f'{name}\n{pool.dataset_name} / {pool.llm_name}', cbar_label=name)

    # b) seed-only vs multi-init ensembles at matched size
    idx, inits, seeds = pool.balanced_subset()
    ens_rows = []
    if len(inits) >= 2 and len(seeds) >= 2:
        lab = {(m['init'], m['seed']): i for i, m in enumerate(pool.members)}
        k = min(len(inits), len(seeds))
        families = {'seed_only_matched': [], 'multi_init_matched': [], 'seed_only_full': [], 'multi_init_full': []}
        for init in inits:
            for combo in itertools.combinations(seeds, k):
                families['seed_only_matched'].append([lab[(init, s)] for s in combo])
            families['seed_only_full'].append([lab[(init, s)] for s in seeds])
        for s in seeds:
            for combo in itertools.combinations(inits, k):
                families['multi_init_matched'].append([lab[(i, s)] for i in combo])
            families['multi_init_full'].append([lab[(i, s)] for i in inits])
        # full-size families are identical to the matched ones when #inits == #seeds
        if len(seeds) == k:
            families.pop('seed_only_full')
        if len(inits) == k:
            families.pop('multi_init_full')

        report.section(f'b) Uniform-average ensembles: seed-only vs multi-init (matched size k={k})')
        report.line(f"{'family':<20}{'n':>4}{'size':>6}{'ens acc':>10}{'gain/mean':>11}{'gain/best':>11}"
                    f"{'avg err':>10}{'ambiguity':>11}{'ens err':>10}")
        fam_acc = {}
        for fam, member_lists in families.items():
            stats = [ensemble_stats(probs, y_onehot, y, torch.as_tensor(m)) for m in member_lists]
            for m, st in zip(member_lists, stats):
                ens_rows.append(dict(info, family=fam, size=len(m), members='|'.join(pool.labels[i] for i in m),
                                     n_distinct_inits=len(set(pool.inits[m])), **st))
            fam_acc[fam] = np.array([st['ens_acc'] for st in stats])
            avg = {key: np.mean([st[key] for st in stats]) for key in stats[0]}
            report.line(f"{fam:<20}{len(stats):>4}{len(member_lists[0]):>6}{avg['ens_acc']:>10.2f}"
                        f"{avg['gain_vs_mean']:>+11.2f}{avg['gain_vs_best']:>+11.2f}{avg['avg_member_sq_err']:>10.4f}"
                        f"{avg['ambiguity']:>11.4f}{avg['ens_sq_err']:>10.4f}")
            summary.update({f'{fam}_{key}': avg[key] for key in ('ens_acc', 'gain_vs_mean', 'gain_vs_best',
                                                                    'ambiguity', 'avg_member_sq_err', 'ens_sq_err')})
            summary[f'{fam}_n'] = len(stats)
        p = mannwhitneyu(fam_acc['multi_init_matched'], fam_acc['seed_only_matched'], alternative='greater').pvalue
        summary['matched_multi_init_vs_seed_only_mwu_p'] = p
        report.line(f'Multi-init > seed-only (matched size, Mann-Whitney one-sided): p={p:.4g}')
        report.line('(ensemble error = avg member error - ambiguity; higher ambiguity at equal member error = '
                    'more useful diversity)')
    else:
        report.line('b) skipped: need >= 2 inits and >= 2 common seeds')

    # c) diversity vs gain over k-subsets
    report.section('c) Diversity vs gain over k-subsets of the pool')
    subset_rows = []
    for k in parse_list(args.subset_sizes, int):
        if k < 2 or k > len(pool):
            continue
        res = subset_sweep(probs, y, acc, pw['disagreement'], pool.inits, k, args.max_subsets, rng)
        rho, p = spearmanr(res['diversity'], res['gain_vs_mean'])
        report.line(f"k={k}: {len(res['subsets'])} subsets{' (sampled of ' + str(res['total']) + ')' if res['sampled'] else ''} "
                    f"| Spearman(diversity, gain/mean) = {rho:.3f} (p={p:.3g})")
        for n in sorted(set(res['n_inits'])):
            m = res['n_inits'] == n
            report.line(f"    {n} distinct init(s): n={m.sum():>6}  diversity {res['diversity'][m].mean() * 100:6.2f}%  "
                        f"ens acc {res['ens_acc'][m].mean():6.2f}  gain/mean {res['gain_vs_mean'][m].mean():+6.2f}  "
                        f"gain/best {res['gain_vs_best'][m].mean():+6.2f}")
            subset_rows.append(dict(info, k=k, n_distinct_inits=int(n), n_subsets=int(m.sum()),
                                    sampled=res['sampled'], spearman_div_gain=rho, spearman_p=p,
                                    mean_diversity=res['diversity'][m].mean(), mean_ens_acc=res['ens_acc'][m].mean(),
                                    mean_gain_vs_mean=res['gain_vs_mean'][m].mean(),
                                    mean_gain_vs_best=res['gain_vs_best'][m].mean(),
                                    max_ens_acc=res['ens_acc'][m].max()))
        summary[f'k{k}_spearman_div_gain'] = rho
        if not args.no_plots:
            plot_div_vs_gain(res, k, os.path.join(out_dir, f'{pool.tag}_diversity_vs_gain_k{k}.png'), pool)

    append_rows([summary], os.path.join(out_dir, 'complementarity_summary.csv'))
    if ens_rows:
        append_rows(ens_rows, os.path.join(out_dir, 'complementarity_ensembles.csv'))
    if subset_rows:
        append_rows(subset_rows, os.path.join(out_dir, 'complementarity_subsets.csv'))
    report.line(f'\nReport: {report.path}')


if __name__ == '__main__':
    import argparse

    parser = add_common_args(argparse.ArgumentParser(
        description='Complementarity of LoRA-init views and ensemble gain (multi-init vs seed-only)'))
    parser.add_argument('--subset_sizes', type=str, default='2,3,4,5',
                        help='Subset sizes k for the diversity-vs-gain sweep')
    parser.add_argument('--max_subsets', type=int, default=20000,
                        help='Sample at most this many k-subsets per k')
    parser.add_argument('--no_plots', action='store_true')
    main(parser.parse_args())
