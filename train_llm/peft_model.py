import os
import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType, PrefixTuningConfig, AdaLoraConfig, IA3Config, PromptTuningConfig, LoftQConfig
from transformers import BitsAndBytesConfig

def get_model_path(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    local_model_dir_path = os.path.join(repo_root, "local_models")
    model_name = model_name
    model_path = os.path.join(local_model_dir_path, model_name)
    return model_path

def load_llm_model(cfg):
    
    model_path = get_model_path(cfg.llm.model_name)
    if cfg.llm.model_name == "llama_3.2_1B":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            num_labels= cfg.dataset.num_classes,
            torch_dtype=torch.bfloat16,
            device_map="cuda", )
    else:
        ## load with quantization 4 bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            num_labels=cfg.dataset.num_classes,
            quantization_config=quantization_config,
            device_map="auto",
        )
        
    return model


def get_lora_model(model, cfg):
    # Prepare loftq_config if using LoftQ initialization
    loftq_config = None
    if cfg.peft.init_lora_weights == 'loftq':
        loftq_config = LoftQConfig(
            loftq_bits=4,  # 4-bit quantization (can be 2, 4, or 8)
            loftq_iter=1,  # number of alternating iterations for quantization and SVD (default: 1)
        )
    
        LoRA_Config = LoraConfig(
            r=cfg.peft.rank,
            lora_alpha=cfg.peft.lora_alpha,
            lora_dropout=cfg.peft.lora_dropout,
            target_modules=cfg.llm.ft_target_modules,  # include v/o
            task_type=TaskType.SEQ_CLS,
            modules_to_save=cfg.peft.module_to_save,  # <-- keep classifier head trainable
            init_lora_weights=cfg.peft.init_lora_weights,
            use_rslora=cfg.peft.use_rslora,
            loftq_config=loftq_config,  # Only used when init_lora_weights='loftq'
        )
    else:
        LoRA_Config = LoraConfig(
            r=cfg.peft.rank,
            lora_alpha=cfg.peft.lora_alpha,
            lora_dropout=cfg.peft.lora_dropout,
            target_modules=cfg.llm.ft_target_modules,  # include v/o
            task_type=TaskType.SEQ_CLS,
            modules_to_save=cfg.peft.module_to_save,  # <-- keep classifier head trainable
            init_lora_weights=cfg.peft.init_lora_weights,
            use_rslora=cfg.peft.use_rslora,
        )
    trainable_model = get_peft_model(model, LoRA_Config)
    return trainable_model

if __name__ == "__main__":
    from config import setup_finetuning_cfg

    cfg = setup_finetuning_cfg(dataset_name="cora", llm_name="llama_3.2_1B", peft_type="lora")
    base_model = load_llm_model(cfg)
    model = get_lora_model(base_model, cfg)
    for name, modules in base_model.named_modules():
        print(name, modules)