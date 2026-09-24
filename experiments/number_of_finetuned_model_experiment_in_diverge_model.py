# in this file we want to run abliation study for my perp
# in diverge we have paraperter name M thats number of initiliztaion apporch and nubmer of fine tuned moded
# and also number of gnn model
# now i want to run it for M=2 TO 5
# first train 5 gnn model with diffrent embedding initilization apporch
# then pich m best gnn model and run ensemble
# you can read for more information my paper in diverge_paper/template.tex
# i want to add this abliation study to my paper in the section \subsubsection{Analysis of the Impact of Number of Finetuned LLM} but i do not want add it in this modemoent to .tex file
# one thing report with reporter and place it int csv file
# run and report with diffirent seed.
"""
Ablation study: impact of M (number of fine-tuned LLM init approaches / GNN models
combined in the DIVERGE ensemble) on downstream performance.

Pipeline for one (dataset, llm, seed):
  1) Train all 5 GNNs, one per LoRA init approach (PISSA, ORTHOGONAL, GAUSSIAN,
     LOFTQ, EVA), on their cached embeddings.
  2) Rank the 5 individual GNNs by validation accuracy (best first).
  3) For each M in {2, 3, 4, 5}: take the M best-ranked models (nested selection,
     so the M=3 set is the M=2 set plus the next-best model, etc. - this keeps the
     ablation a clean "adding models helps" story instead of an M-dependent
     reshuffle) and run ensemble learning restricted to just those M models.
  4) Report every individual + per-M ensemble result via the shared reporter and
     append one row per (dataset, llm, seed, M) to a CSV report, so the numbers
     for the paper's "Analysis of the Impact of Number of Finetuned LLM"
     subsubsection (diverge_paper/template.tex, currently left empty on purpose)
     can be aggregated/plotted across seeds later.

Not written to diverge_paper/template.tex - CSV + reporter output only, as requested.
"""
from datetime import datetime

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import gnn_train_and_report, append_gnn_result_csv
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter
from report.reporter import ReportResults

# Order must match dataset.data_utils.get_init_dataset_for_gnn's return tuple
APPROACHES = ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']
TITLES = {'pissa': 'PISSA', 'orthogonal': 'ORTHOGONAL', 'gaussian': 'GAUSSIAN', 'loftq': 'LOFTQ', 'eva': 'EVA'}


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         ensemble_approaches='learnable', m_values=None, reporter_index=0,
         csv_path='results/ablation/num_finetuned_models_results.csv'):
    """
    Run the M-ablation study for one (dataset, llm, seed) combination.

    Args:
        seed: Seed the LoRA adapters / embedding caches were trained with (must match
              --seed used in train_llm/train_other_lora_init_apporaches.py and
              cache/cache_embedding_with_diffrent_init_weights.py); also used as the
              GNN's own training seed. If None, uses configs/dataset/<name>.json's seed.
        supervised: True for the supervised embedding cache/GNN split, False for the
              semi-supervised (seed-tagged) one.
        ensemble_approaches: weight-learning approach passed to the ensemble
              (see ensemble.ensemble_gnns_learning.ensemble_gnn_learning).
        m_values: which M values to ablate over (default: 2..5, i.e. all subset
              sizes up to using all 5 fine-tuned models).
        csv_path: where to append the complete-info result row ("" / None disables it).

    Returns:
        dict with the 5 individual results, the val-accuracy ranking, and the
        per-M ensemble results/hyperparameters.
    """
    m_values = m_values or [2, 3, 4, 5]
    if any(m < 1 or m > len(APPROACHES) for m in m_values):
        raise ValueError(f"m_values must be within [1, {len(APPROACHES)}], got {m_values}")

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed

    print("=" * 80)
    print(f"ABLATION: Number of Finetuned LLMs (M) | dataset={dataset_name} llm={llm_name} "
          f"peft={peft_type} seed={resolved_seed} supervised={supervised} M={m_values}")
    print("=" * 80)

    reporter = ReportResults(
        cfg,
        save_dir=f"results/train_info/{dataset_name}_num_finetuned_ablation_seed{resolved_seed}",
        index_run=reporter_index,
    )
    reporter.report_title("Ablation: Impact of Number of Finetuned LLMs (M)")
    reporter.report_txt(f"Dataset: {dataset_name}")
    reporter.report_txt(f"LLM: {llm_name}")
    reporter.report_txt(f"PEFT Type: {peft_type}")
    reporter.report_txt(f"Seed: {resolved_seed}")
    reporter.report_txt(f"Supervised: {supervised}")
    reporter.report_txt(f"Ensemble Approach: {ensemble_approaches}")
    reporter.report_txt(f"M values tested: {m_values}")
    reporter.report_txt("")

    # 1) Load the seed-tagged embeddings and train all 5 GNNs (each call also
    #    appends its own row to results/gnn_training/gnn_training_results.csv)
    data_by_approach = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=resolved_seed)
    all_data = dict(zip(APPROACHES, data_by_approach))
    emb = {name: get_embedding_from_data(all_data[name]) for name in APPROACHES}
    print("Loaded embeddings: " + ", ".join(f"{name}={tuple(emb[name].shape)}" for name in APPROACHES))

    gnn_kwargs = dict(does_print_training_process=False, reporter=reporter, supervised=supervised,
                       seed=resolved_seed, llm_name=llm_name, peft_type=peft_type)
    results, models = {}, {}
    for name in APPROACHES:
        results[name], models[name] = gnn_train_and_report(dataset_name, emb[name], title=TITLES[name], **gnn_kwargs)

    # 2) Rank the 5 individual GNNs by validation accuracy (best first)
    ranked = sorted(APPROACHES, key=lambda n: results[n]['val_acc'], reverse=True)
    reporter.report_title("Individual Model Ranking (by validation accuracy)")
    for rank, name in enumerate(ranked, 1):
        line = (f"  {rank}. {TITLES[name]}: val_acc={results[name]['val_acc']:.4f}, "
                f"test_acc={results[name]['test_acc']:.4f}, test_f1={results[name]['test_f1']:.4f}")
        print(line)
        reporter.report_txt(line)

    # 3) For each M, ensemble the M best-ranked models (nested top-M selection)
    m_results = {}
    for m in m_values:
        chosen = ranked[:m]
        chosen_titles = [TITLES[n] for n in chosen]
        print(f"\n{'=' * 80}\nM={m} | Ensembling: {chosen_titles}\n{'=' * 80}")

        ensemble_results, ensemble_params, _ = tune_ensemble_hyperparameter(
            dataset_name,
            [models[n] for n in chosen],
            [emb[n] for n in chosen],
            does_report_training_process=False,
            ensemble_approaches=ensemble_approaches,
            title=f"M={m} Ablation (seed={resolved_seed})",
            reporter=reporter,
            llm_name=llm_name, peft_type=peft_type,
            supervised=supervised, seed=resolved_seed,
            visualize_best=False,
        )
        m_results[m] = {"chosen": chosen, "results": ensemble_results, "params": ensemble_params}

        # 4a) Report via the shared reporter
        reporter.report_title(f"M={m} - Ensemble of {chosen_titles}")
        reporter.report_txt(f"Chosen models (best-{m} by val accuracy): {chosen_titles}")
        reporter.report_txt(f"Train - Acc: {ensemble_results['train']['accuracy']:.4f}, "
                             f"Macro-F1: {ensemble_results['train']['macro_f1']:.4f}")
        reporter.report_txt(f"Val   - Acc: {ensemble_results['val']['accuracy']:.4f}, "
                             f"Macro-F1: {ensemble_results['val']['macro_f1']:.4f}")
        reporter.report_txt(f"Test  - Acc: {ensemble_results['test']['accuracy']:.4f}, "
                             f"Macro-F1: {ensemble_results['test']['macro_f1']:.4f}, "
                             f"Weighted-F1: {ensemble_results['test']['weighted_f1']:.4f}")
        reporter.report_txt(f"Best ensemble hyperparameters: {ensemble_params}")
        reporter.report_txt("")

        print(f"[M={m}] test_acc={ensemble_results['test']['accuracy']:.4f} "
              f"test_f1={ensemble_results['test']['macro_f1']:.4f} params={ensemble_params}")

        # 4b) Append complete-info CSV row
        if csv_path:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_name": dataset_name,
                "llm_name": llm_name,
                "peft_type": peft_type,
                "seed": resolved_seed,
                "supervised": supervised,
                "ensemble_weight_approach": ensemble_approaches,
                "M": m,
                "chosen_approaches": ",".join(chosen_titles),
                "ensemble_num_layers": ensemble_params.get("num_layers", ""),
                "ensemble_hidden_dim": ensemble_params.get("hidden_dim", ""),
                "ensemble_dropout": ensemble_params.get("dropout", ""),
                "ensemble_learning_rate": ensemble_params.get("learning_rate", ""),
                "train_accuracy": ensemble_results['train']['accuracy'],
                "train_macro_f1": ensemble_results['train']['macro_f1'],
                "train_weighted_f1": ensemble_results['train']['weighted_f1'],
                "val_accuracy": ensemble_results['val']['accuracy'],
                "val_macro_f1": ensemble_results['val']['macro_f1'],
                "val_weighted_f1": ensemble_results['val']['weighted_f1'],
                "test_accuracy": ensemble_results['test']['accuracy'],
                "test_macro_f1": ensemble_results['test']['macro_f1'],
                "test_weighted_f1": ensemble_results['test']['weighted_f1'],
            }
            append_gnn_result_csv(row, csv_path)

    if csv_path:
        print(f"[OK] Ablation results appended to: {csv_path}")

    # Summary across M, for a quick console/report readout of the trend
    reporter.report_title("Summary: Ensemble Test Performance vs M")
    summary_lines = []
    for m in m_values:
        r = m_results[m]["results"]
        line = f"  M={m}: Test Acc={r['test']['accuracy']:.4f}, Test Macro-F1={r['test']['macro_f1']:.4f}"
        summary_lines.append(line)
        reporter.report_txt(line)
    print("\n" + "\n".join(summary_lines))

    return {
        "individual_results": results,
        "ranking": ranked,
        "m_results": m_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ablation study: impact of M (number of fine-tuned LLM init approaches) on DIVERGE's ensemble")
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed of the trained LoRA adapters/embedding caches (must match --seed used in '
                              'train_llm/train_other_lora_init_apporaches.py and the caching step)')
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embeddings/GNN split, 0 = semi-supervised (seed-tagged) split')
    parser.add_argument('--ensemble_approaches', type=str, default='learnable')
    parser.add_argument('--m_values', type=str, default='2,3,4,5',
                         help='Comma-separated list of M values to ablate over (each in [1, 5])')
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/ablation/num_finetuned_models_results.csv')
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        ensemble_approaches=args.ensemble_approaches,
        m_values=[int(m.strip()) for m in args.m_values.split(',') if m.strip()],
        reporter_index=args.reporter_index,
        csv_path=args.csv_path,
    )
