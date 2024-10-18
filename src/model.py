import os
from dotenv import load_dotenv
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from .utils.common import load_config, load_csv_data
from .utils.dataset import TextDataset
from .utils.callback import CustomEarlyStoppingCallback, CustomSaveModelCallback
import shutil
from loguru import logger
from datetime import datetime
import pandas as pd

os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
os.environ["WANDB_MODE"] = "dryrun"  # offline mode

load_dotenv()
model_name = os.getenv('MODEL_NAME')

print(model_name)

class Model_GPT2:
    def __init__(self, model_name=model_name):
        print ("Model name: ", model_name)
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Đặt pad token là eos token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        special_tokens_dict = {'sep_token': '<|sep|>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loading_dataset(self, filename):
        df = load_csv_data(filename)
        # df['text'] = df['Question'] + " " + df['Answer']
        df = self.various_dataset(df)
        df['text'] = df['Question'] + " <|sep|> " + df['Answer']

        texts_raw = df['text'].tolist()
        self.clean_text(texts_raw)

        texts = texts_raw * 5

        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
        sep_token_id = self.tokenizer.convert_tokens_to_ids('<|sep|>')
        # dataset = TextDataset(encodings)
        dataset = TextDataset(encodings, sep_token_id)
        return dataset
    
    def clean_text(self, list_text):
        for i in range(len(list_text)):
            if (list_text[i][-1] != "."):
                list_text[i] = list_text[i] + "."
    
    def various_dataset(self, pd_raw):
        new_question = []
        new_answer = []
        for i in range(len(pd_raw)):
            new_question.append(pd_raw.iloc[i]['Question'].lower())
            new_answer.append(pd_raw.iloc[i]['Answer'])
            new_question.append(pd_raw.iloc[i]['Question'].upper())
            new_answer.append(pd_raw.iloc[i]['Answer'])
        new_data = {
            'Question': new_question,
            'Answer': new_answer,
        }
        new_rows_df = pd.DataFrame(new_data)
        return pd.concat([pd_raw, new_rows_df], ignore_index=True)

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

    def train(self, train_dataset, lr=5e-4, epochs=4):
        self.clear_cache()

        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        logger.add(f"./logs/gpt2_{time}.log")

        # custom_callback = CustomEarlyStoppingCallback(threshold=0.0000001, logger=logger)
        custom_callback = CustomEarlyStoppingCallback(logger=logger)
        save_model_callback = CustomSaveModelCallback(self.model, self.tokenizer, epoch_save=1, gpt_model=self)


        training_args = TrainingArguments(
            output_dir='./results',           # output directory
            logging_dir='./logs',             # directory for storing logs
            report_to="wandb",                # directory for storing logs
            run_name="wandb_chat_gpt2",       # name of the W&B run (optional)
            logging_steps=1,                  # how often to log to W&B
            num_train_epochs=epochs,          # Total number of training epochs
            per_device_train_batch_size=2,    # Batch size per device during training
            save_steps=10_000,                # After how many steps to save the model
            save_total_limit=2,               # Limit the total amount of checkpoints
            prediction_loss_only=True,
            learning_rate=lr
        )

        trainer = Trainer(
            model=self.model,                 # the instantiated Transformers model to be trained
            args=training_args,               # training arguments, defined above
            train_dataset=train_dataset,      # training dataset
            callbacks=[custom_callback, save_model_callback]
        )
        trainer.train()

    def save(self, path='./models'):
        directory = os.path.abspath(path)
        if os.path.exists(directory):
            shutil.rmtree(directory)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def clear_cache(self):
        directories = ['./results', './models', './wandb', './logs']
        for directory in directories:
            path = os.path.abspath(directory)
            if os.path.exists(directory):
                shutil.rmtree(path)

def run_model(config_file):
    return load_config(config_file)

