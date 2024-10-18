import os
from src.model import Model_GPT2

Model = Model_GPT2()
dataset = Model.loading_dataset(os.path.abspath("./data/data_test.xlsx"))

Model.train(dataset)
Model.save()
