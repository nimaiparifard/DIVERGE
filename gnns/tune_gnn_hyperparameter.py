from gnns.gnn_mtrainer import set_mgnn_cfg, GNNTrainer, get_datasets_path
import itertools
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


def get_candidate_grid_search_hyperparamters():
    candidate_hyperparameters = {
        "learning_rate": [0.01, 0.005, 0.001, 0.0001],
        "hidden_dim": [128, 256, 512, 1024],
        "num_layers": [2, 3],
        "dropout": [0.0, 0.1, 0.01, 0.5],
        "weight_decay": [0.0, 0.01],
        "batch_norm": [0, 1],

    }
    return candidate_hyperparameters

def tune_gnn_hyperparamters(dataset_name, features, device, supervised=True):
    """
        features in init features torch base (num_nodes, feature_dim)
        load dataset with load_graph_dataset_for_tape
        for training gnn use GNNTrainer
        setting the hyperparamter in cfg for example cfg.lr =
    """
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised)
    cfg.device = 1
    
    # Get candidate hyperparameters
    candidate_hyperparams = get_candidate_grid_search_hyperparamters()
    
    # Generate all combinations of hyperparameters
    keys = list(candidate_hyperparams.keys())
    values = list(candidate_hyperparams.values())
    combinations = list(itertools.product(*values))
    
    print(f"Total hyperparameter combinations to try: {len(combinations)}")
    
    best_val_acc = 0.0
    best_hyperparameters = None
    best_results = None
    
    for idx, combination in enumerate(combinations):
        print(f"\n{'='*80}")
        print(f"Testing combination {idx+1}/{len(combinations)}")
        
        # Create hyperparameter dict for this combination
        current_hyperparams = dict(zip(keys, combination))
        print(f"Hyperparameters: {current_hyperparams}")
        
        # Update cfg with current hyperparameters
        cfg.lr = current_hyperparams["learning_rate"]
        cfg.hidden_dim = current_hyperparams["hidden_dim"]
        cfg.num_layers = current_hyperparams["num_layers"]
        cfg.dropout = current_hyperparams["dropout"]
        cfg.weight_decay = current_hyperparams["weight_decay"]
        cfg.batch_norm = bool(current_hyperparams["batch_norm"])
        
        try:
            # Train with current hyperparameters
            trainer = GNNTrainer(cfg, features)
            results, _, history = trainer.train()
            
            print(f"Results - Val Acc: {results['val_acc']:.4f}, Test Acc: {results['test_acc']:.4f}")
            
            # Visualize learning curve for this hyperparameter combination
            visualize_learning_curve(dataset_name, current_hyperparams, history)
            
            # Update best hyperparameters if current is better
            if results['val_acc'] > best_val_acc:
                best_val_acc = results['val_acc']
                best_hyperparameters = current_hyperparams.copy()
                best_results = results.copy()
                print(f"*** New best validation accuracy: {best_val_acc:.4f} ***")
        except Exception as e:
            print(f"Error with combination {idx+1}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Best hyperparameters found:")
    print(f"Hyperparameters: {best_hyperparameters}")
    print(f"Best Val Acc: {best_results['val_acc']:.4f}")
    print(f"Best Test Acc: {best_results['test_acc']:.4f}")
    print(f"Best Val F1: {best_results['val_f1']:.4f}")
    print(f"Best Test F1: {best_results['test_f1']:.4f}")
    
    return best_hyperparameters, best_results

def save_hyperparameters(dataset_name, best_hyperparameters, re_split=1):
    """
        Save the best result in gnn_hyperparameters/{datset_name}.json
    """
    # Create directory if it doesn't exist
    current_dir = Path(__file__).resolve().parent
    save_dir = current_dir / "gnn_hyperparameters"
    save_dir.mkdir(exist_ok=True)
    
    # Prepare the hyperparameters to save
    hyperparams_to_save = {
        "seed": 45,
        "device": 1,
        "dataset": dataset_name,
        "gnn_model_name": "SAGE",
        "llm_name": "llama_3.2_1B",
        "epochs": 200,
        "early_stop": 20,
        "re_split": re_split,
        "lr": best_hyperparameters["learning_rate"],
        "hidden_dim": best_hyperparameters["hidden_dim"],
        "num_layers": best_hyperparameters["num_layers"],
        "dropout": best_hyperparameters["dropout"],
        "weight_decay": best_hyperparameters["weight_decay"],
        "batch_norm": best_hyperparameters["batch_norm"],
    }
    
    # Save to JSON file
    save_path = save_dir / f"{dataset_name}.json"
    if re_split == 0:
        save_path = save_dir / f"{dataset_name}_semi_supervised.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams_to_save, f, indent=4)
    
    print(f"\nBest hyperparameters saved to: {save_path}")

def visualize_learning_curve(dataset_name, hyperparameters, history, save_dir="experiment_results"):
    """
        visualize the learning curve of the gnn model for testing hyperparameters and save the plot in experiment_results/{dataset_name}/{}.png in the name of plot specifiy the hyperparameters in the name of the plot
        return the learning curve
    """
    # Create save directory
    current_dir = Path(__file__).resolve().parent
    save_path = current_dir / save_dir / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with hyperparameters
    filename = f"lr{hyperparameters['learning_rate']}_hd{hyperparameters['hidden_dim']}_nl{hyperparameters['num_layers']}_do{hyperparameters['dropout']}_wd{hyperparameters['weight_decay']}_bn{hyperparameters['batch_norm']}.png"
    full_path = save_path / filename
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_acc'], 'g-', linewidth=2, label='Train Acc', alpha=0.7)
    axes[1].plot(epochs, history['val_acc'], 'b-', linewidth=2, label='Val Acc')
    axes[1].plot(epochs, history['test_acc'], 'r-', linewidth=2, label='Test Acc')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    axes[2].plot(epochs, history['train_f1'], 'g-', linewidth=2, label='Train F1', alpha=0.7)
    axes[2].plot(epochs, history['val_f1'], 'b-', linewidth=2, label='Val F1')
    axes[2].plot(epochs, history['test_f1'], 'r-', linewidth=2, label='Test F1')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title(f'F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Add hyperparameters as suptitle
    hyperparams_str = f"lr={hyperparameters['learning_rate']}, hidden_dim={hyperparameters['hidden_dim']}, layers={hyperparameters['num_layers']}, dropout={hyperparameters['dropout']}, weight_decay={hyperparameters['weight_decay']}, batch_norm={hyperparameters['batch_norm']}"
    fig.suptitle(f'Learning Curves - {dataset_name}\n{hyperparams_str}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curve saved to: {full_path}")
    
    return history
    
if __name__ == "__main__":
    import torch
    from common import load_graph_dataset_for_tape
    import argparse
    
    ## argument definition
    parser = argparse.ArgumentParser(description='Tune GNN hyperparameters')
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        help='Dataset name (e.g., cora, citeseer, pubmed, wikics)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0). If not specified, will auto-detect')
    parser.add_argument('--re_split', type=int, default=0,
                        help='Re-split value (0 for semi-supervised, >0 for supervised)')
    
    args = parser.parse_args()
    
    # Example usage
    dataset_name = args.dataset_name
    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else -1
    else:
        device = args.device
    print("Device:", device)
    re_split = args.re_split
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
    # Load dataset to get initial features
    print(f"Loading dataset: {dataset_name}")
    cfg = set_mgnn_cfg(dataset_name)
    data, num_classes, text = load_graph_dataset_for_tape(dataset_name, device, re_split=re_split, path_prefix=path_prefix, seed=cfg.seed)
    if re_split > 0:
        features = torch.load(f'artifacts/cache/llama_3.2_1B_{dataset_name}_seqcls_lora_init-pissa_pool-mean.pt')['embeddings']
    else:
        features = torch.load(f'../../../artifacts/cache/llama_3.2_1B_{dataset_name}_seqcls_lora_init-pissa_pool-mean_semi_supervised.pt')['embeddings']
    print(f"Dataset loaded - Nodes: {features.shape[0]}, Features: {features.shape[1]}, Classes: {num_classes}")
    
    # Run hyperparameter tuning
    print(f"\nStarting hyperparameter tuning for {dataset_name}...")
    best_hyperparameters, best_results = tune_gnn_hyperparamters(dataset_name, features, device, supervised=False)
    
    # # Save the best hyperparameters
    save_hyperparameters(dataset_name, best_hyperparameters, re_split=re_split)
    #
    # print(f"\n{'='*80}")
    # print(f"Hyperparameter tuning completed!")
    # print(f"Best results saved in gnn_hyperparameters/{dataset_name}.json")
    # print(f"Learning curves saved in experiment_results/{dataset_name}/")