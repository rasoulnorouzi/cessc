# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig 
from scripts.configurations import config
import torch
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType
)
# %%
def initialize_model(model_name):

    """
        Initialize and return the model and tokenizer based on the provided model name.
    
        Args:
            model_name (str): The name of the model to initialize (e.g., "BERT", "SciBERT", "Roberta").

        Returns:
            tuple: A tuple containing the initialized model and tokenizer.
    """
    if model_name=="BERT":
        model = AutoModelForSequenceClassification.from_pretrained(config["BERT"]["model_settings"]["model_type"], num_labels=config["BERT"]["model_settings"]["num_labels"])
        tokenizer = AutoTokenizer.from_pretrained(config["BERT"]["model_settings"]["model_type"])
    elif model_name=="SciBERT":
        model = AutoModelForSequenceClassification.from_pretrained(config["SciBERT"]["model_settings"]["model_type"], num_labels=config["SciBERT"]["model_settings"]["num_labels"])
        tokenizer = AutoTokenizer.from_pretrained(config["SciBERT"]["model_settings"]["model_type"])
    elif model_name=="Roberta":
        model = AutoModelForSequenceClassification.from_pretrained(config["Roberta"]["model_settings"]["model_type"], num_labels=config["Roberta"]["model_settings"]["num_labels"])
        tokenizer = AutoTokenizer.from_pretrained(config["Roberta"]["model_settings"]["model_type"])
    
    elif model_name=="LLAMA2-7b":
        tokenizer = AutoTokenizer.from_pretrained(
            config["LLAMA2-7b"]["model_settings"]["model_type"],
            token = config["LLAMA2-7b"]["token_pass"]
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["LLAMA2-7b"]["quantization"]["load_in_4bit"],
            load_4bit_use_double_quant=config["LLAMA2-7b"]["quantization"]["load_4bit_use_double_quant"],
            bnb_4bit_quant_type=config["LLAMA2-7b"]["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config["LLAMA2-7b"]["model_settings"]["model_type"],
            num_labels=config["LLAMA2-7b"]["model_settings"]["num_labels"],
            quantization_config=bnb_config,
            device_map=config["LLAMA2-7b"]["device_map"],
            token = config["LLAMA2-7b"]["token_pass"]
        )

        model.config.pad_token_id = model.config.eos_token_id

        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config["LLAMA2-7b"]["peft_config"]["r"], 
            lora_alpha=config["LLAMA2-7b"]["peft_config"]["lora_alpha"], 
            lora_dropout=config["LLAMA2-7b"]["peft_config"]["lora_dropout"],
            bias=config["LLAMA2-7b"]["peft_config"]["bias"],
            target_modules=config["LLAMA2-7b"]["peft_config"]["target_modules"],
            inference_mode=False
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        model.config.use_cache = False

    elif model_name=="Mistral-7b":
        tokenizer = AutoTokenizer.from_pretrained(config["Mistral-7b"]["model_settings"]["model_type"],
                                                  token = config["Mistral-7b"]["token_pass"])
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["Mistral-7b"]["quantization"]["load_in_4bit"],
            load_4bit_use_double_quant=config["Mistral-7b"]["quantization"]["load_4bit_use_double_quant"],
            bnb_4bit_quant_type=config["Mistral-7b"]["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config["Mistral-7b"]["model_settings"]["model_type"],
            num_labels=config["Mistral-7b"]["model_settings"]["num_labels"],
            quantization_config=bnb_config,
            device_map=config["Mistral-7b"]["device_map"],
            token = config["Mistral-7b"]["token_pass"]
        )

        model.config.pad_token_id = model.config.eos_token_id

        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config["Mistral-7b"]["peft_config"]["r"], 
            lora_alpha=config["Mistral-7b"]["peft_config"]["lora_alpha"], 
            lora_dropout=config["Mistral-7b"]["peft_config"]["lora_dropout"],
            bias=config["Mistral-7b"]["peft_config"]["bias"],
            target_modules=config["Mistral-7b"]["peft_config"]["target_modules"],
            inference_mode=False
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False
        

    return model, tokenizer
