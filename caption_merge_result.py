# This is a sample Python script.
import argparse
import json
import os
import re
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import threading
import traceback

import PIL.Image as Image
import io
import pandas as pd
import torch
from tqdm import tqdm

import checkpoint_utils
from transformers import pipeline, AutoTokenizer
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER, \
    IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.run_llava import load_images
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

num = "cap3"


# model_path = "./checkpoints/llava-1.5-7b-hf-task-lora"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_finetune,
#     model_base=None,
#     model_name=get_model_name_from_path(model_finetune)
# )

# model_path = "bczhou/TinyLLaVA-3.1B"
# prompt = "What are the things I should be cautious about when I visit here?"
# image_file = "https://llava-vl.github.io/static/images/view.jpg"

# "Give you 2 examples: \n" \
# "1. `Contradiction. The image shows earth covered with snow, with a silhouette of a baby covered in a warm blanket evoking the warmth and care of a mother's embrace, which is the opposite of feeling exposed and vulnerable.`\n" \
# "2. `Entailment. The image displays RoboCop, a character from a science fiction film who is a police officer brought back to life as a cyborg after his death. The text above reads: \"DUDE DIED, BUT THEY MADE HIM GO TO WORK ANYWAY.\" This entails the claim that even death won't exempt you from going to work because it humorously illustrates a character who has been reanimated as a cyborg to continue working despite having died.`\n" \
def train(data, image_folder):
    file_name = "gemini"
    result = []
    data = checkpoint_utils.load_checkpoint("result/{}/_{}.json".format(num, file_name))
    caption_test = checkpoint_utils.load_checkpoint("data/caption_test.json")
    for index, row in tqdm(enumerate(data), total=len(data)):
        row["explanation"] = caption_test[index]["description"] + row["explanation"]
        result.append({
            "id": index,
            "label": row["label"],
            "explanation": row["explanation"],
        })
    checkpoint_utils.save_checkpoint(result,
                                     "result/{}/new_{}.json".format(num, file_name))

    r = pd.DataFrame(result)

    # 将DataFrame保存为CSV文件
    csv_filename = "result/{}/new_{}.csv".format(num, file_name)
    r.to_csv(csv_filename, index=False)


def get_data(path):
    df = pd.read_parquet(path)
    return df


def main():
    file = "./data/test_ac.json"
    data = json.load(open(file))
    image_folder = "../data/fl/"
    # part1 = get_data("../data/fl/test-00000-of-00002.parquet")
    # part2 = get_data("../data/fl/test-00001-of-00002.parquet")
    # final_part = part1 + part2
    # 使用concat追加part2到part1
    # final_part = pd.concat([part1, part2], ignore_index=True)

    train(data, image_folder)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
