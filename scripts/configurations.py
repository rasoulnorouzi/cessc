from transformers import TrainingArguments

seed_value = 8642

config = {
    "BERT": {
        "model_settings": {
            "model_type": "bert-base-uncased",
            "num_labels": 2
        },
        "training_args": TrainingArguments(
            output_dir= "output_BERT",
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",  # This line is added to match the evaluation_strategy
            logging_steps=1,
            weight_decay=0.1,
            learning_rate=2e-5,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=seed_value
)

    },

    "SciBERT": {
        "model_settings": {
            "model_type": "allenai/scibert_scivocab_uncased",
            "num_labels": 2
        },
        "training_args": TrainingArguments(

            output_dir= "output",
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",  # This line is added to match the evaluation_strategy
            logging_steps=1,
            weight_decay=0.1,
            learning_rate=2e-5,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=seed_value
)
    },

    "Roberta": {
        "model_settings": {
            "model_type": "roberta-large",
            "num_labels": 2
        },

        "training_args": TrainingArguments(
            
            output_dir= "output",
            num_train_epochs=10,
            per_device_train_batch_size=20,
            per_device_eval_batch_size=20,
            evaluation_strategy="epoch",
            save_strategy="epoch",  # This line is added to match the evaluation_strategy
            logging_steps=1,
            weight_decay=0.1,
            learning_rate=2e-5,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=seed_value
)
    },

    "LLAMA2-7b": {
        "model_settings": {
            "model_type": "meta-llama/Llama-2-7b-hf",
            "num_labels": 2
        },
        "device_map":"auto",
        "token_pass":"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",
        "quantization":
        {
        "load_in_4bit":"True",
        "load_4bit_use_double_quant":"True",
        "bnb_4bit_quant_type":"nf4",
        "bnb_4bit_compute_dtype":"torch.bfloat16",
        },
        "peft_config":{
            "r":8, 
            "lora_alpha":32, 
            "lora_dropout":0.1,
            "bias":"none",
            "target_modules":["q_proj","v_proj"],
        },

        "training_args": TrainingArguments(
            output_dir= "LLAMA2-7b_output",
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",  # This line is added to match the evaluation_strategy
            logging_steps=1,
            weight_decay=0.01,
            learning_rate=2e-5,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=seed_value,
            fp16=False

        )
    },
    "Mistral-7b": {
        "model_settings": {
            "model_type": "mistralai/Mistral-7B-v0.1",
            "num_labels": 2
        },
        "device_map":"auto",
        "token_pass":"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",
        "quantization": 
        {
        "load_in_4bit":"True",
        "load_4bit_use_double_quant":"True",
        "bnb_4bit_quant_type":"nf4",
        "bnb_4bit_compute_dtype":"torch.bfloat16",
        },
        "peft_config":{
            "r":8, 
            "lora_alpha":32, 
            "lora_dropout":0.1,
            "bias":"none",
            "target_modules":["q_proj","v_proj"],
        },
        "training_args": TrainingArguments(
            output_dir= "output",
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",  # This line is added to match the evaluation_strategy
            logging_steps=1,
            weight_decay=0.01,
            learning_rate=2e-5,
            do_train=True,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=seed_value,
            fp16=False
            )   
        }
    }