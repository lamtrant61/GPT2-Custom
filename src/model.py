import os
from dotenv import load_dotenv
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from .utils.common import load_config

load_dotenv()
model_name = os.getenv('MODEL_NAME')

print(model_name)
class Model_GPT2:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, x):
        return x

def run_model(config_file):
    return load_config(config_file)

