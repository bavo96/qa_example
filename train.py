from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from datasets import load_dataset
import data
import config
import os 
import torch
from evaluation import compute_metrics
import sys

if __name__ == "__main__":
    # SET ENVIRONMENT
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_list = ['bert', 'roberta', 'bartpho']
    model_name = model_list[1]

    if model_name == 'roberta':
        url = "xlm-roberta-base"
    elif model_name == 'bert':
        url = "deepset/bert-base-cased-squad2"
    elif model_name == 'bartpho':
        url = "vinai/bartpho-syllable"

    ## LOAD CONFIG
    cfg = config.Config(
            train_data_path="data_qa_law_train.csv",
            test_data_path="data_qa_law_test.csv",
            model_url=url,  
            output_dir="trained_model",
            saved_model_path="trained_model/final_model/",
            optim="adamw_torch",
            test_size=0.2,
            train_size=0.8,
            learning_rate=2e-5,
            weight_decay=0.01,
            batch_size=8,
            n_epoch=100,
            use_cpu=False,
            max_length=512,
            stride=128,
            )
    print(cfg)

    # LOAD DATA
    train_dataset = load_dataset("csv", data_files=cfg.train_data_path, split="all")
    test_dataset = load_dataset("csv", data_files=cfg.test_data_path, split="all")

    # TOKENIZE DATA
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_url, is_fast=False)
    print(tokenizer)
    print(type(tokenizer))

    map_train_dataset = train_dataset.map(data.preprocess_training_data, 
                                  batched=True, 
                                  remove_columns=train_dataset.column_names,
                                  fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length, "stride": cfg.stride})
    map_test_dataset = test_dataset.map(data.preprocess_validation_data,
                                             batched=True,
                                             remove_columns=test_dataset.column_names,
                                             fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.max_length, "stride": cfg.stride})
    
    print(map_test_dataset)
    # LOAD PRETRAINED MODEL
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_url, ignore_mismatched_sizes=True) 
    # print(model)
    # print(type(model))

    # SET TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        optim=cfg.optim,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.n_epoch,
        save_strategy="no",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        report_to="tensorboard",
        push_to_hub=False,
        use_cpu=cfg.use_cpu,
    )

    # CREATE TRAINER
    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=map_train_dataset,
        eval_dataset=map_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # TRAIN MODEL
    trainer.train()

    # SAVE FINAL MODEL
    trainer.save_model(cfg.saved_model_path)





