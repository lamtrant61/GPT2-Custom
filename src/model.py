import os
from dotenv import load_dotenv
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from .utils.common import load_config, load_csv_data
from .utils.dataset import TextDataset
from .utils.callback import CustomEarlyStoppingCallback
import shutil
# from tqdm import tqdm
# from datasets import Dataset
# import pandas as pd

os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
os.environ["WANDB_MODE"] = "dryrun"  # offline mode

load_dotenv()
model_name = os.getenv('MODEL_NAME')

print(model_name)

class Model_GPT2:
    def __init__(self, model_name=model_name):
        print ("Model name: ", model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Đặt pad token là eos token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loading_dataset(self, filename):
        df = load_csv_data(filename)
        df['text'] = df['Question'] + " <|sep|> " + df['Answer']

        texts_raw = df['text'].tolist()
        texts = texts_raw * 5

        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
        dataset = TextDataset(encodings)
        return dataset

    def preprocess_data(self, data):
        return self.tokenizer(data, padding="max_length", truncation=True, max_length=512)
    
    def generate(self, data, config=None):
        # Preprocess the input data
        input_text = data + " <|sep|> "
        input_ids = self.preprocess_data(input_text)
        input_ids = torch.tensor(input_ids["input_ids"]).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        
        # Generate text using the model
        output = self.model.generate(
            input_ids,
            pad_token_id = self.tokenizer.pad_token_id,  # Set pad token ID
            **config
        )
        
        # Decode the generated output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.replace("<|sep|>", "").strip()

    def train(self, train_dataset, lr=5e-4, epochs=10):
        self.clear_cache()
        custom_callback = CustomEarlyStoppingCallback(threshold=0.0001)
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            report_to="wandb",            # directory for storing logs
            run_name="wandb_chat_gpt2",  # name of the W&B run (optional)
            logging_steps=1,  # how often to log to W&B
            num_train_epochs=epochs,               # Total number of training epochs
            per_device_train_batch_size=2,    # Batch size per device during training
            save_steps=10_000,                 # After how many steps to save the model
            save_total_limit=2,                # Limit the total amount of checkpoints
            prediction_loss_only=True,
            learning_rate=lr
        )

        trainer = Trainer(
            model=self.model,                # the instantiated Transformers model to be trained
            args=training_args,              # training arguments, defined above
            train_dataset=train_dataset,      # training dataset
            callbacks=[custom_callback]
        )
        trainer.train()

    def save(self, path='./models'):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def clear_cache(self):
        directories = ['./results', './models']
        for directory in directories:
            path = os.path.abspath(directory)
            if os.path.exists(directory):
                shutil.rmtree(path)
        
        

def run_model(config_file):
    return load_config(config_file)

