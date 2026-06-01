from train_llm.peft_model import load_llm_model, get_lora_model
from dataset.dataset_loader import load_dataset, get_tokenizer
from config import setup_finetuning_cfg
from train_llm.train import Train, load_llm_with_lora_adapter, evaluate_sampled_examples
from adapter_dir import get_semi_supervised_adapter_dir
from report.reporter import ReportResults

def main(database_name, llm_name, peft_type, init_weight_approach, split):
    cfg = setup_finetuning_cfg(dataset_name=database_name, llm_name=llm_name, peft_type=peft_type)
    init_weights_approaches_list =  ['pissa', 'gaussian', 'eva',  'loftq', 'orthogonal']
    # init_weights_approaches_list = ['orthogonal']
    cfg.peft.init_lora_weights = init_weight_approach  # change this to try different initialization approaches
    print("=== Using LoRA initialization approach:", cfg.peft.init_lora_weights)
    # Train with LLM responses
    adapter_dir = get_semi_supervised_adapter_dir(cfg)
    save_dir = f'results/train_info/train_1/{database_name}_{llm_name}_{peft_type}_{init_weight_approach}_semi_supervised_adapter'
    reporter = ReportResults(cfg, save_dir=save_dir)
    train = Train(cfg, adapter_dir, reporter, split=split)
    train.process()


if __name__ == "__main__":
    ## argument definition
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with different LoRA initialization approaches')
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        help='Dataset name (e.g., cora, citeseer, pubmed, wikics)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        help='LLM model name')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='PEFT type (e.g., lora)')
    parser.add_argument('--init_weight_approach', type=str, default='gaussian',
                        choices=['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal'],
                        help='LoRA initialization approach')
    parser.add_argument('--split', type=int, default=0,)
    
    args = parser.parse_args()
    
    # Run main function with parsed arguments
    main(
        database_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        init_weight_approach=args.init_weight_approach,
        split=args.split
    )
