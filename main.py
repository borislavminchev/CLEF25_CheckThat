import nltk
nltk.download('punkt')

import torch
import os
import evaluate
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    print("GPU is enabled.")
    print("device count: {}, current device: {}".format(torch.cuda.device_count(), torch.cuda.current_device()))
else:
    print("GPU is not enabled.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #make sure GPU is enabled.

import accelerate
import transformers
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer

from nltk.tokenize import RegexpTokenizer
from torch.utils.data import DataLoader

print(transformers.__version__) #4.47.1
print(accelerate.__version__) #1.2.1


# Load Model and Tokenzier

model_checkpoint = 'google/umt5-base'
model_code = model_checkpoint.split("/")[-1]
metric = evaluate.load("meteor")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

config = AutoConfig.from_pretrained(
    model_checkpoint,
    max_length=128,
    length_penalty=0.6,
    no_repeat_ngram_size=2,
    num_beams=15,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=config).to(device)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    return_tensors="pt")

print(model_checkpoint)

#Prepare Data

train_data = pd.read_csv("train/train-eng.csv") # change path and file name 
val_data = pd.read_csv("dev/dev-eng.csv")

train_data = train_data.sample(frac=1).reset_index(drop=True)
val_data = val_data.sample(frac=1).reset_index(drop=True)

ds = DatasetDict({
        'train': Dataset.from_pandas(train_data),
        'validation': Dataset.from_pandas(val_data)
        })

def tokenize_sample_data(data):
    # Max token size is set to 1024 and 128 for inputs and labels, respectively.
    input_feature = tokenizer(data["text"], truncation=True, max_length=1024)
    label = tokenizer(data["claim"], truncation=True, max_length=128)
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }


tokenized_ds = ds.map(
    tokenize_sample_data,
    remove_columns=["claim", "text"],
    batched=True,
    batch_size=1)


def tokenize_sentence(arg):
    encoded_arg = tokenizer(arg)
    return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

def metrics_func(eval_arg):
    preds, labels = eval_arg
    # Replace -100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Convert id tokens to text
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Insert a line break (\n) in each sentence for scoring
    text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
    sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]
    # compute METEOR score with custom tokenization
    return metric.compute(
        predictions=text_preds,
        references=text_labels,
        tokenizer=tokenize_sentence
    )

# Training
training_args = Seq2SeqTrainingArguments(
    output_dir = f"saved-models-{model_code}",
    num_train_epochs = 1,  # epochs
    learning_rate = 5e-4,
    lr_scheduler_type = "linear",
    warmup_steps = 90,
    optim = "adamw_hf",
    weight_decay = 0.01,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 16,
    eval_steps = 100,
    predict_with_generate=True,
    generation_max_length = 128,
    logging_steps = 10,
    push_to_hub = False
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    compute_metrics = metrics_func,
    train_dataset = tokenized_ds["train"],
    eval_dataset = tokenized_ds["validation"],
    tokenizer = tokenizer
)

trainer.train()

os.makedirs(f"{model_code}/finetuned_{model_code}", exist_ok=True)

if hasattr(trainer.model, "module"):
    trainer.model.module.save_pretrained(f"./{model_code}/finetuned_{model_code}")
else:
    trainer.model.save_pretrained(f"./{model_code}/finetuned_{model_code}")

print("Training done")

#Inference


# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(f"./{model_code}/finetuned_{model_code}")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

input_text = "This is some sample text."

# Tokenize the Input Text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    generated_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True)

# Decode the Generated Output
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Print the Output
print(f"Generated Output: {output_text}")