from typing import List, Union
import random
import os
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm


prompts = ["Here are some samples of a dataset:\n\n",
            "\nAccording to the format, distribution, content and column name and column relation of the above samples"\
    ", as well as your logic and general knowledge, please generate 10 new data in the same format directly. "\
            "The new samples must not exist in the given data. The samples should also conform to the internal logic "\
            "in feature combination. Please do not use codes to generate samples, but give the generated samples"\
            " directly based on your knowledge."""]

with open("./data/development.json", "r") as fp:
    ori_data = fp.readlines()

random.shuffle(ori_data)
i=90
prompt = "".join(ori_data[i:i+10])
prompts = prompts[0] + prompt + prompts[1]
print(prompts)