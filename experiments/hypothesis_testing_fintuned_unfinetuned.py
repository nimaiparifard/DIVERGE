import os

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data, get_based_model_dataset
from gnns.gnn_mtrainer import get_datasets_path, gnn_train_and_report
from report.reporter import ReportResults
from common import load_graph_dataset_for_tape
from statsmodels.stats.contingency_tables import mcnemar
import torch


def _resolve_path_prefix():
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            return '.'
        return os.path.normpath(os.path.relpath(repo_root, start=os.getcwd()))
    return '../..'


def _select_embedding_by_title(title, cfg):
    normalized = title.strip().lower().replace(' ', '_')
    if normalized in {'based', 'base', 'based_model'}:
        return get_embedding_from_data(get_based_model_dataset(cfg))

    data_pissa, data_orthogonal, data_loftq, data_eva, data_guassian = get_init_dataset_for_gnn(cfg)
    title_to_data = {
        'pissa': data_pissa,
        'orthogonal': data_orthogonal,
        'loftq': data_loftq,
        'eva': data_eva,
        'gaussian': data_guassian,
        'guassian': data_guassian,
    }
    if normalized not in title_to_data:
        valid = "based_model, pissa, orthogonal, loftq, eva, gaussian"
        raise ValueError(f"Unsupported model title '{title}'. Supported values: {valid}")
    return get_embedding_from_data(title_to_data[normalized])


def run_hypthesis_testing_per_sample(based_gnn_model, finetuned_gnn_model, dataset_name, cfg, titles, finetuned_features, base_features):
    # run hypthesis testing to show fintuning llm get statiscallcy better result for without fintuning model
    # run McNemar Test
    # bring this mcnemart test for test data
    base_title = titles[0] if len(titles) > 0 else 'based_model'
    finetuned_title = titles[1] if len(titles) > 1 else 'pissa'

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    path_prefix = _resolve_path_prefix()

    data, _, _ = load_graph_dataset_for_tape(
        dataset_name,
        device,
        re_split=1,
        path_prefix=path_prefix,
        seed=cfg.dataset.seed,
    )

    y = data.y.squeeze().to(device)
    test_mask = data.test_mask
    edge_index = data.edge_index

    # base_features = _select_embedding_by_title(base_title, cfg).to(device)
    # finetuned_features = _select_embedding_by_title(finetuned_title, cfg).to(device)

    based_gnn_model = based_gnn_model.to(device)
    finetuned_gnn_model = finetuned_gnn_model.to(device)
    based_gnn_model.eval()
    finetuned_gnn_model.eval()

    with torch.no_grad():
        base_pred = based_gnn_model(base_features, edge_index).argmax(dim=1)
        finetuned_pred = finetuned_gnn_model(finetuned_features, edge_index).argmax(dim=1)

    y_test = y[test_mask]
    base_correct = (base_pred[test_mask] == y_test)
    finetuned_correct = (finetuned_pred[test_mask] == y_test)

    both_correct = int((base_correct & finetuned_correct).sum().item())
    base_only_correct = int((base_correct & (~finetuned_correct)).sum().item())
    finetuned_only_correct = int(((~base_correct) & finetuned_correct).sum().item())
    both_wrong = int(((~base_correct) & (~finetuned_correct)).sum().item())

    contingency_table = [
        [both_correct, base_only_correct],
        [finetuned_only_correct, both_wrong],
    ]

    discordant = base_only_correct + finetuned_only_correct
    use_exact = discordant < 25
    mcnemar_result = mcnemar(contingency_table, exact=use_exact, correction=not use_exact,)

    alpha = 0.05
    is_significant = mcnemar_result.pvalue < alpha
    finetuned_better = finetuned_only_correct > base_only_correct

    summary = {
        'titles': [base_title, finetuned_title],
        'contingency_table': contingency_table,
        'discordant_pairs': discordant,
        'mcnemar_statistic': float(mcnemar_result.statistic),
        'p_value': float(mcnemar_result.pvalue),
        'alpha': alpha,
        'significant': bool(is_significant),
        'finetuned_better_on_discordant': bool(finetuned_better),
        'conclusion': (
            f"{finetuned_title} is significantly better than {base_title}"
            if is_significant and finetuned_better
            else f"No significant evidence that {finetuned_title} outperforms {base_title}"
        ),
    }

    print("\n" + "=" * 80)
    print(f"McNemar per-sample test on TEST split ({base_title} vs {finetuned_title})")
    print(f"Contingency table [[both_correct, base_only_correct], [finetuned_only_correct, both_wrong]]: {contingency_table}")
    print(f"Discordant pairs (b + c): {discordant}")
    print(f"McNemar statistic: {summary['mcnemar_statistic']:.6f}")
    print(f"p-value: {summary['p_value']:.6g} (alpha={alpha})")
    print(f"Conclusion: {summary['conclusion']}")
    print("=" * 80 + "\n")

    return summary

if __name__ == '__main__':
    dataset_name = 'cora'

    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    reporter_index = dataset_name + "_" + "error_rate"
    reporter = ReportResults(cfg, index_run=reporter_index)
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        # If running from project root, use current directory
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            path_prefix = '.'
        else:
            # Otherwise, use relative path from current directory to repo root
            path_prefix = os.path.relpath(repo_root, start=os.getcwd())
            # Normalize path to avoid double slashes
            path_prefix = os.path.normpath(path_prefix)
    else:
        # Fallback to default
        path_prefix = '../..'

    dataset, _, _ = load_graph_dataset_for_tape(dataset_name, 'cuda:0', re_split=1, path_prefix=path_prefix,
                                                seed=cfg.dataset.seed)
    data_pissa, data_orthogonal, data_loftq, data_eva, data_guassian = get_init_dataset_for_gnn(cfg)
    emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian = get_embedding_from_data(
        data_pissa), get_embedding_from_data(data_orthogonal), get_embedding_from_data(
        data_loftq), get_embedding_from_data(data_eva), get_embedding_from_data(data_guassian)
    results_pissa, model_pissa = gnn_train_and_report(dataset_name, emb_pissa, title="PISSA",
                                                      does_print_training_process=False, reporter=reporter)
    results_orthogonal, model_orthogonal = gnn_train_and_report(dataset_name, emb_orthogonal, title="ORTHOGONAL",
                                                                does_print_training_process=False, reporter=reporter)
    results_loftq, model_loftq = gnn_train_and_report(dataset_name, emb_loftq, title="LOFTQ",
                                                      does_print_training_process=False, reporter=reporter)
    results_eva, model_eva = gnn_train_and_report(dataset_name, emb_eva, title="EVA", does_print_training_process=False,
                                                  reporter=reporter)
    results_guassian, model_guassian = gnn_train_and_report(dataset_name, emb_guassian, title="GUASSIAN",
                                                            does_print_training_process=False, reporter=reporter)


    based_model_data = get_based_model_dataset(cfg)

    based_model_data_emb = get_embedding_from_data(based_model_data)

    results_based_model, model_based_model = gnn_train_and_report(dataset_name, based_model_data_emb, title="BASED MODEL",
                                                  does_print_training_process=False, reporter=reporter)
    titles = ["Based_Model", "Pissa"]
    run_hypthesis_testing_per_sample(model_based_model, model_eva, dataset_name, cfg, titles, emb_eva, based_model_data_emb)
