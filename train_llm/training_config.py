from transformers import TrainingArguments

def get_training_config(cfg):
    training_args = TrainingArguments(
        output_dir=cfg.training_args.output_dir,
        overwrite_output_dir=cfg.training_args.overwrite_output_dir,
        num_train_epochs=cfg.training_args.num_train_epochs,
        per_device_train_batch_size=cfg.training_args.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training_args.per_device_eval_batch_size,
        learning_rate=cfg.training_args.learning_rate,
        gradient_accumulation_steps=cfg.training_args.gradient_accumulation_steps,
        max_grad_norm=cfg.training_args.max_grad_norm,
        weight_decay=cfg.training_args.weight_decay,
        lr_scheduler_type=cfg.training_args.lr_scheduler_type,
        warmup_ratio=cfg.training_args.warmup_ratio,
        logging_dir=cfg.training_args.logging_dir,
        logging_steps=cfg.training_args.logging_steps,
        eval_strategy=cfg.training_args.eval_strategy,
        eval_steps=cfg.training_args.eval_steps,
        save_strategy=cfg.training_args.save_strategy,
        save_steps=cfg.training_args.save_steps,
        metric_for_best_model=cfg.training_args.metric_for_best_model,
        load_best_model_at_end=cfg.training_args.load_best_model_at_end,
        fp16=cfg.training_args.fp16,
        bf16=cfg.training_args.bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=cfg.remove_unused_columns,
        report_to=cfg.report_to,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        use_cpu=cfg.use_cpu,
    )
    return training_args