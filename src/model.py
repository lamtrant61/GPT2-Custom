import os
from dotenv import load_dotenv
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from .utils.common import load_config, load_csv_data
import tensorflow as tf
from tqdm import tqdm
from datasets import Dataset
import pandas as pd

load_dotenv()
model_name = os.getenv('MODEL_NAME')

print(model_name)
class Model_GPT2:
    def __init__(self, model_name=model_name):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Đặt pad token là eos token
        self.model = TFGPT2LMHeadModel.from_pretrained(model_name, from_pt=True)
        self.model.summary()

    def loading_dataset(self, filename):
        df = load_csv_data(filename)
        df['text'] = df['Question'] + " " + df['Answer']

        # Apply tokenizer to all text
        tokenized_data = df['text'].apply(self.preprocess_data)
        tokenized_df = pd.DataFrame(tokenized_data.tolist())
        return Dataset.from_pandas(tokenized_df)

    def preprocess_data(self, data):
        return self.tokenizer(data, padding="max_length", truncation=True, max_length=512)
    
    def generate(self, data):
        input_ids = self.preprocess_data(data)
        input_ids = tf.constant(input_ids["input_ids"])
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def train(self, data, lr=5e-5, epochs=3):
        training_args = TrainingArguments(
            output_dir='../results',          # output directory
            num_train_epochs=epochs,         # number of training epochs
            per_device_train_batch_size=4,   # batch size for training
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='../logs',            # directory for storing logs
            logging_steps=10,
            learning_rate=lr
        )

        trainer = Trainer(
            model=self.model,                # the instantiated Transformers model to be trained
            args=training_args,              # training arguments, defined above
            train_dataset=train_dataset      # training dataset
        )
        trainer.train()

    def save(self, path='../models'):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        

def run_model(config_file):
    return load_config(config_file)

