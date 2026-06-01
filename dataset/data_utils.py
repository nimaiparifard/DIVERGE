"""
Utility functions for loading and processing data used across multiple modules.
This module helps avoid circular imports between edge_editing.py and ensemble_gnns_learning.py
"""
import torch
import os


def get_init_dataset_for_gnn(cfg, supervised=True):
    """
    Load initial datasets for GNN training with different initialization methods.
    
    Args:
        cfg: Configuration object with dataset and model settings
        supervised: If True, load supervised datasets; if False, load semi-supervised datasets
        
    Returns:
        tuple: (data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva)
    """
    from gnns.gnn_mtrainer import get_datasets_path

    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)

    # Try multiple possible cache directory locations
    cache_dir = None
    possible_cache_dirs = [
        os.path.join(repo_root, 'artifacts', 'cache'),
        os.path.join(repo_root, 'datasets', 'cache'),
        os.path.abspath('artifacts/cache'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'cache')),
    ]
    
    for possible_dir in possible_cache_dirs:
        if os.path.exists(possible_dir):
            cache_dir = possible_dir
            break
    
    if cache_dir is None:
        raise FileNotFoundError(
            f"Cannot find cache directory. Tried:\n" +
            "\n".join(f"  - {d}" for d in possible_cache_dirs) +
            f"\n\nCurrent working directory: {os.getcwd()}\n"
            f"Repository root: {repo_root}"
        )
    
    # Normalize path to avoid issues with trailing slashes
    cache_dir = os.path.normpath(cache_dir)
    
    # Build file paths based on supervised flag, with fallback to base names
    suffix = '_semi_supervised' if not supervised else ''
    base_name = f'{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}'
    
    init_types = {
        'pissa': 'pissa',
        'orthogonal': 'orthogonal',
        'loftq': 'loftq',
        'eva': 'eva',
        'guassian': 'gaussian',
    }
    
    data_paths = {}
    for key, init_name in init_types.items():
        suffixed = os.path.join(cache_dir, f'{base_name}_init-{init_name}_pool-mean{suffix}.pt')
        base = os.path.join(cache_dir, f'{base_name}_init-{init_name}_pool-mean.pt')
        data_paths[key] = suffixed if os.path.exists(suffixed) else base
    
    data_path_pissa = data_paths['pissa']
    data_path_orthogonal = data_paths['orthogonal']
    data_path_loftq = data_paths['loftq']
    data_path_eva = data_paths['eva']
    data_path_guassian = data_paths['guassian']
    
    # Check if files exist before loading
    missing_files = []
    for name, path in data_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"  - {name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required cache files:\n" + "\n".join(missing_files) +
            f"\n\nCache directory: {cache_dir}\n"
            f"Supervised mode: {supervised}\n"
            f"Please ensure these files exist or generate them first."
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load all datasets
    try:
        data_pissa = torch.load(data_path_pissa, map_location=device)
        data_orthogonal = torch.load(data_path_orthogonal, map_location=device)
        data_eva = torch.load(data_path_eva, map_location=device)
        data_loftq = torch.load(data_path_loftq, map_location=device)
        data_guassian = torch.load(data_path_guassian, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Error loading cache files from {cache_dir}:\n"
            f"  {str(e)}\n"
            f"Please verify the files are valid PyTorch checkpoint files."
        ) from e
    
    return data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva

def get_cached_dir():
    """
    Get the cache directory path, trying multiple possible locations.

    Returns:
        str: Absolute path to the cache directory

    Raises:
        FileNotFoundError: If no cache directory is found
    """
    from gnns.gnn_mtrainer import get_datasets_path
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)

    # Try multiple possible cache directory locations
    possible_cache_dirs = [
        os.path.join(repo_root, 'artifacts', 'cache'),
        os.path.join(repo_root, 'datasets', 'cache'),
        os.path.abspath('artifacts/cache'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'cache')),
    ]
    
    for possible_dir in possible_cache_dirs:
        if os.path.exists(possible_dir):
            return os.path.normpath(possible_dir)
    
    raise FileNotFoundError(
        f"Cannot find cache directory. Tried:\n" +
        "\n".join(f"  - {d}" for d in possible_cache_dirs) +
        f"\n\nCurrent working directory: {os.getcwd()}\n"
        f"Repository root: {repo_root}"
    )

def get_init_dataset_for_gnn_with_retrained_gnn_mistake(cfg):
    """
    Load datasets for GNN training with different initialization methods from retrained models.
    These models were retrained on GNN mistakes for improved performance.
    
    Args:
        cfg: Configuration object with dataset and model settings
        
    Returns:
        tuple: (data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva)
    """
    cache_dir = get_cached_dir()
    
    # Build file paths for retrained models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path_pissa = os.path.join(cache_dir, f'{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init-pissa_pool-mean.pt')
    data_path_orthogonal = os.path.join(cache_dir, f'{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init-orthogonal_pool-mean.pt')
    data_path_loftq = os.path.join(cache_dir, f'{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init-loftq_pool-mean.pt')
    data_path_eva = os.path.join(cache_dir, f'{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init-eva_pool-mean.pt')
    data_path_guassian = os.path.join(cache_dir, f'{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init-gaussian_pool-mean.pt')
    
    # Check if files exist before loading
    missing_files = []
    for name, path in [
        ('pissa', data_path_pissa),
        ('orthogonal', data_path_orthogonal),
        ('loftq', data_path_loftq),
        ('eva', data_path_eva),
        ('guassian', data_path_guassian)
    ]:
        if not os.path.exists(path):
            missing_files.append(f"  - {name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required cache files for retrained models:\n" + "\n".join(missing_files) +
            f"\n\nCache directory: {cache_dir}\n"
            f"Please ensure these files exist or generate them first."
        )
    
    # Load all datasets for retrained models
    try:
        data_pissa = torch.load(data_path_pissa, map_location=device)
        data_orthogonal = torch.load(data_path_orthogonal, map_location=device)
        data_eva = torch.load(data_path_eva, map_location=device)
        data_loftq = torch.load(data_path_loftq, map_location=device)
        data_guassian = torch.load(data_path_guassian, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Error loading cache files from {cache_dir}:\n"
            f"  {str(e)}\n"
            f"Please verify the files are valid PyTorch checkpoint files."
        ) from e
    
    return data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva

def get_based_model_dataset(cfg):
    """
    Load dataset from base model (without PEFT adapters).
    
    Args:
        cfg: Configuration object with dataset and model settings
        
    Returns:
        Data object from base model
        
    Raises:
        FileNotFoundError: If the cache file doesn't exist
        RuntimeError: If there's an error loading the file
    """
    cached_dir = get_cached_dir()

    data_based_path = os.path.join(cached_dir, f"{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_pool-mean_based_model.pt")

    if not os.path.exists(data_based_path):
        raise FileNotFoundError(
            f"Base model cache file not found: {data_based_path}\n"
            f"Cache directory: {cached_dir}\n"
            f"Please ensure this file exists or generate it first."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        based_model_data = torch.load(data_based_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Error loading base model cache file from {data_based_path}:\n"
            f"  {str(e)}\n"
            f"Please verify the file is a valid PyTorch checkpoint file."
        ) from e

    return based_model_data

def get_embedding_from_data(data):
    """
    Extract embeddings from a data object.
    
    Args:
        data: Data object containing embeddings (dict or similar structure)
        
    Returns:
        torch.Tensor: The embeddings
    """
    return data['embeddings']
