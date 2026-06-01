import torch
from time import time
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from dataset.data_utils import get_init_dataset_for_gnn
from common import GNNEncoder, compute_acc_and_f1
from common import load_graph_dataset_for_tape, set_seed
from yacs.config import CfgNode as CN
import json
from pathlib import Path

def get_datasets_path():
    """Get the absolute path to the datasets folder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    datasets_path = os.path.join(repo_root, "datasets")
    return datasets_path

def set_mgnn_cfg(datset_name='cora', supervised=True):
    """
    Build a yacs CfgNode for GNN training. This function will try to load
    a JSON hyperparameter file from either the local `gnn_hyperparameters/`
    folder (preferred) or the legacy `hyperparameters/` folder.

    The returned cfg contains fields consumed by `GNNTrainer`:
      - seed, device, dataset, gnn_model_name, llm_name
      - hidden_dim, num_layers, dropout, lr, epochs, early_stop
      - batch_norm, weight_decay, re_split

    If no JSON is found, sensible defaults are used.
    """

    # Find candidate paths for the hyperparameter JSON
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / 'gnn_hyperparameters' / f"{datset_name}.json",
    ]
    if not supervised:
        candidates = [
            repo_root / 'gnn_hyperparameters' / f"{datset_name}_semi_supervised.json",
        ]

    hp = {}
    for c in candidates:
        if c.exists():
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    hp = json.load(f)
                print(f"Loaded GNN hyperparameters from: {c}")
                break
            except Exception as e:
                print(f"Failed to load {c}: {e}")

    # Defaults
    defaults = {
        'seed': 42,
        'device': 0,
        'dataset': datset_name,
        'gnn_model_name': 'SAGE',
        'llm_name': 'llama_3.2_1B',
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'lr': 0.05,
        'epochs': 200,
        'early_stop': 20,
        'batch_norm': 1,
        'weight_decay': 0.1,
        're_split': 1,
    }

    # Merge loaded hyperparams with defaults
    for k, v in defaults.items():
        defaults[k] = hp.get(k, v)

    cfg = CN()
    cfg.seed = int(defaults['seed'])
    cfg.device = int(defaults['device'])
    cfg.dataset = defaults['dataset']
    cfg.gnn_model_name = defaults['gnn_model_name']
    cfg.llm_name = defaults['llm_name']
    cfg.hidden_dim = int(defaults['hidden_dim'])
    cfg.num_layers = int(defaults['num_layers'])
    cfg.dropout = float(defaults['dropout'])
    cfg.lr = float(defaults['lr'])
    cfg.epochs = int(defaults['epochs'])
    cfg.early_stop = int(defaults['early_stop'])
    cfg.batch_norm = bool(int(defaults['batch_norm']))
    cfg.weight_decay = float(defaults['weight_decay'])
    cfg.re_split = bool(int(defaults.get('re_split', 1)))

    return cfg

class GNNTrainer():
    def __init__(self, cfg, feature, modified_dataset=None, does_print_training_process=False):
        self.seed = cfg.seed
        set_seed(cfg.seed)
        self.device = torch.device("cuda:0" if cfg.device > 0 else "cpu")
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn_model_name
        self.lm_model_name = cfg.llm_name
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.dropout = cfg.dropout
        self.lr = cfg.lr
        self.features = feature
        self.epochs = cfg.epochs
        self.patience = cfg.early_stop
        self.batch_norm = cfg.batch_norm
        self.weight_decay = cfg.weight_decay
        self.does_print_training_process = does_print_training_process
        self.error_rate = dict()
        self.error_number = dict()
        self.best_model = None
        # Load data
        set_seed(self.seed)
        # Get absolute path to project root (where datasets folder is located)
        datasets_path = get_datasets_path()
        repo_root = os.path.dirname(datasets_path)
        # Use absolute path or relative path from current working directory
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
        data, num_classes, _ = load_graph_dataset_for_tape(cfg.dataset, self.device, re_split=cfg.re_split, path_prefix=path_prefix, seed=self.seed, modified_dataset=modified_dataset, )

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        train_size = train_mask.sum().item()
        val_size = val_mask.sum().item()
        test_size = test_mask.sum().item()
        print(f"train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")
        re_split_prefix, re_split_suffix = '_s_' if cfg.re_split else '', f'-seed{self.seed}'
        # Init gnn feature

        self.model = GNNEncoder(
            input_dim=self.features.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=int(data.y.max().item() + 1),
            n_layers=self.num_layers,
            gnn_type=self.gnn_model_name,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)


        self.features = self.features.to(self.device)
        self.data = data.to(self.device)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of GNN parameters: {trainable_params}")
        self.ckpt = f"../../results/TAPE/{self.dataset_name}/{self.gnn_model_name}.pt"

    def _train(self):
        # the same in ../main.py where we train GNNs
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(self.features, self.data.edge_index)
        loss = F.cross_entropy(output[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def _evaluate(self):
        # the same in ../main.py where we evaluate GNNs
        self.model.eval()
        logits = self.model(self.features, self.data.edge_index)
        pred = logits.argmax(dim=1)
        accuracy, macrof1_scores, weightf1_scores = [], [], []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            acc, macro_f1, weight_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), self.data.y[mask].cpu().numpy())

            accuracy.append(acc)
            macrof1_scores.append(macro_f1)
            weightf1_scores.append(weight_f1)

        # Per-class error stats on the **test** split
        test_mask = self.data.test_mask
        error_rate, error_number = self.compute_error_rate(
            pred[test_mask].cpu().numpy(),
            self.data.y[test_mask].cpu().numpy()
        )
        return accuracy, macrof1_scores, weightf1_scores, logits, error_rate, error_number


    def compute_error_rate(self, pred, y):
        """
            Compute per-class error statistics on a given split.

            Args:
                pred (np.ndarray): predicted class indices for the split
                y (np.ndarray): ground-truth class indices for the split

            Returns:
                error_rate (dict):   {class_id: error_rate_percentage}
                error_number (dict): {class_id: number_of_misclassified_examples}
        """
        error_rate = {}
        error_number = {}

        # Ensure numpy arrays
        pred = np.asarray(pred)
        y = np.asarray(y)

        # Compute stats for every class present in the dataset
        for class_id in range(self.num_classes):
            class_mask = (y == class_id)
            total = int(class_mask.sum())

            if total == 0:
                # No examples of this class in the evaluated split
                error_number[class_id] = 0
                error_rate[class_id] = 0.0
                continue

            mistakes = int((pred[class_mask] != y[class_mask]).sum())
            error_number[class_id] = mistakes
            # Store percentage for easier interpretation
            error_rate[class_id] = float(mistakes) / float(total) * 100.0

        # Also store on the trainer for external inspection
        self.error_rate = error_rate
        self.error_number = error_number

        return error_rate, error_number

    def plot_learning_curves(self, history, save_dir="../../results/TAPE/learning_curves", show=True):
        """
        Plot and save learning curves for loss, accuracy, and F1 scores.

        Args:
            history: Dictionary containing training history with keys:
                - 'loss': list of training losses
                - 'train_acc', 'val_acc', 'test_acc': lists of accuracies
                - 'train_f1', 'val_f1', 'test_f1': lists of F1 scores
            save_dir: Directory to save the plots
            show: Whether to display the plot (default True)
        """
        os.makedirs(save_dir, exist_ok=True)

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

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(save_dir, f'{self.dataset_name}_seed{self.seed}.png')
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        print(f"Learning curves saved to: {save_path}")
        plt.close()

    def train(self, show_plots=False):
        """
        Train the GNN model.
        
        Args:
            show_plots: Whether to display plots (default False to avoid interruption during hyperparameter tuning)
        """
        # ! Training
        best_eval_acc = best_test_acc = 0.0
        best_eval_f1 = best_test_f1 = 0.0
        timer, counter, best_logits = [], 0, None

        # Initialize history tracking
        history = {
            'loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': [],
            'train_f1': [], 'val_f1': [], 'test_f1': [],
            # Per-epoch test-set error statistics (list of dicts)
            'error_rate': [],
            'error_number': []
        }

        for epoch in range(1, 1 + self.epochs):
            loss = self._train()
            accuracy, f1_scores, weightf1_scores, cur_logits, error_rate, error_number = self._evaluate()

            train_acc, val_acc, test_acc = accuracy
            train_f1, val_f1, test_f1 = f1_scores
            train_weightf1, val_weightf1, test_weightf1 = weightf1_scores

            # Track metrics
            history['loss'].append(loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            history['test_f1'].append(test_f1)
            history['error_rate'].append(error_rate)
            history['error_number'].append(error_number)

            if val_acc > best_eval_acc:
                best_eval_acc = val_acc
                best_test_acc = test_acc
                counter = 0
                best_logits = deepcopy(cur_logits)
                best_eval_f1, best_test_f1 = val_f1, test_f1
                best_eval_weightf1, best_test_weightf1 = val_weightf1, test_weightf1
                best_error_rate, best_error_number = error_rate, error_number
                self.best_model = self.model
            else:
                counter += 1

            if epoch % 10 == 0 and self.does_print_training_process:
                print(
                    f"Epoch {epoch:03d} Loss {loss:.4f}  Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_f1:.3f} Val F1 {val_f1:.3f} Test F1 {test_f1:.4f}")

            # Early stopping
            if counter >= self.patience:
                break

        print(f'\nTest Acc {best_test_acc:.3f}  Test F1 {best_test_f1:.3f}\n')
        self.model = self.best_model
        # Plot and save learning curves (don't show during hyperparameter tuning)
        self.plot_learning_curves(history, show=show_plots)

        return {
            "test_acc": best_test_acc,
            "test_f1": best_test_f1,
            "val_acc": best_eval_acc,
            "val_f1": best_eval_f1,
            "val_weight_f1": best_eval_weightf1,
            "test_weight_f1": best_test_weightf1,
            "error_rate": best_error_rate,
            "error_number": best_error_number,
        }, best_logits, history

def gnn_train_and_report(dataset_name, embedding, title="", does_print_training_process=False, reporter=None, supervised=True):
    """
    Train a GNN model on the given dataset using provided embeddings and report results.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'cora', 'pubmed')
        embedding (torch.Tensor): Node embeddings to use as features
        title (str): Title for reporting (e.g., 'PISSA', 'ORTHOGONAL')
        does_print_training_process (bool): Whether to print training progress
        reporter: Reporter object for writing results to file

    Returns:
        dict: Training results containing test_acc, test_f1, val_acc, val_f1
    """
    print(f"\n{'='*80}")
    print(f"Training GNN on {dataset_name.upper()} dataset")
    print(f"Embedding shape: {embedding.shape}")
    print(f"{'='*80}\n")

    # Setup GNN configuration for the dataset
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised)

    # Initialize GNN trainer with the embeddings
    trainer = GNNTrainer(cfg, embedding, does_print_training_process=does_print_training_process)

    # Train the model and get results
    results, best_logits, history = trainer.train()

    # Print final results
    print(f"\n{'='*80}")
    print(f"Final Results for {dataset_name.upper()}:")
    print(f"  Test Accuracy:       {results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {results['val_f1']:.4f}")
    print(f"  Val F1 (Weighted):   {results['val_weight_f1']:.4f}")
    print(f"{'='*80}\n")

    # Report results using reporter
    if reporter is not None:
        report_title = f"GNN Training Results - {title}" if title else f"GNN Training Results - {dataset_name.upper()}"
        report_text = f"""
                        Final Results:
                        • Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']:.2%})
                        • Test F1 (Macro): {results['test_f1']:.4f}
                        • Test F1 (Weighted): {results['test_weight_f1']:.4f}
                        • Validation Accuracy: {results['val_acc']:.4f} ({results['val_acc']:.2%})
                        • Validation F1 (Macro): {results['val_f1']:.4f}
                        • Validation F1 (Weighted): {results['val_weight_f1']:.4f}
                        """
        reporter.report(report_title, report_text)

    return results, trainer.model

def gnn_train_and_report_with_modified_dataset(modified_dataset, embedding, dataset_name, supervised=True):
    """
    Train a GNN model on the given modified dataset using provided embeddings and report results.

    Args:
        modified_dataset: Modified dataset object with updated edge_index
        embedding (torch.Tensor): Node embeddings to use as features
        dataset_name (str): Name of the dataset (e.g., 'cora', 'pubmed')

    Returns:
        dict: Training results containing test_acc, test_f1, val_acc, val_f1
        model: Trained GNN model
    """
    print(f"\n{'='*80}")
    print(f"Training GNN on {dataset_name.upper()} dataset (MODIFIED)")
    print(f"Embedding shape: {embedding.shape}")
    print(f"{'='*80}\n")

    # Setup GNN configuration for the dataset
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised)

    # Initialize GNN trainer with the embeddings and modified dataset
    trainer = GNNTrainer(cfg, embedding, modified_dataset=modified_dataset)

    # Train the model and get results
    results, best_logits, history = trainer.train()

    # Print final results
    print(f"\n{'='*80}")
    print(f"Final Results for {dataset_name.upper()} (MODIFIED):")
    print(f"  Test Accuracy:       {results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {results['val_f1']:.4f}")
    print(f"  Val F1 (Weighted):   {results['val_weight_f1']:.4f}")
    print(f"{'='*80}\n")

    return results, trainer.model

if __name__ == '__main__':
    print("*" * 80)
    print("Check Hyperparamters")
    gnn_cfg = set_mgnn_cfg('pubmed', supervised=False)
    data_pissa = get_init_dataset_for_gnn(supervised=False)
    print(f"Learning rate: {cfg.lr}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Dropout: {cfg.dropout}")
    print(f"Num Layers: {cfg.num_layers}")

