import os
from src.model import Model_GPT2, run_model
from src.utils.common import load_csv_data

config = run_model(os.path.abspath("./src/config/config.json"))
model_generate_config = config["model_generate_config"]

Model = Model_GPT2()
dataset = Model.loading_dataset(os.path.abspath("./data/data_test.xlsx"))

Model.train(dataset)
Model.save()
