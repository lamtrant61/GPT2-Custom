import os
from src.model import Model_GPT2, run_model
from src.utils.common import load_csv_data

config = run_model(os.path.abspath("./src/config/config.json"))
model_generate_config = config["model_generate_config"]

Model = Model_GPT2(os.path.abspath("./models"))

question = "giới thiệu wghn"
predict = Model.generate(question, model_generate_config)
print ("\n\n\n", predict)

