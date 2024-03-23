# This is a sample Python script.
import json
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


def get_data(path):
    df = pd.read_parquet(path)
    return df


def process_data(df, caption_data, split):
    file_name = "data/{}_ac2.json".format(split)
    result = checkpoint_utils.load_checkpoint(file_name)
    skip_rows = len(result)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if index < skip_rows:
            continue  # 跳过剩余的代码，继续下一个迭代
        caption = caption_data[index]
        claim = row["claim"]
        if "label" in row:
            label = row["label"]
            explanation = row["explanation"]
        else:
            label = 0
            explanation = ""

        prompt1 = "The description of this picture is `{}`. " \
                  "The claim of this picture is `{}`. " \
                  "You need to predict the claim of this picture is 'entailment' or 'contradiction' firstly " \
                  "according to the picture and its description. " \
                  "Then you need to give a explanation for the prediction. " \
                  "The prediction and the explanation are related to the meaning of the figurative language " \
                  "expression. " \
                  "Your response must follow the format shown as below: \n" \
                  "`Prediction. Explanation.`\n".format(caption["description"], claim)
        # 使用 BytesIO 将字节数据转换为文件类对象
        # image_stream = io.BytesIO(image_bytes["bytes"])

        # 使用 Image.open 读取这个对象
        # image = Image.open(image_stream)

        image_path = "image/{}/{}.png".format(split, index)

        # image.save(image_path)

        input = {"from": "human", "value": "<image>\n{}".format(prompt1)}

        output = {
            "from": "gpt",
            "value": "{}. {}".format(label, explanation)
        }

        result.append({
            "id": index,
            "image": image_path,
            "conversations": [input, output],
        })

    checkpoint_utils.save_checkpoint(result,
                                     file_name)


def caption_merge():
    split = ["train", "test"]

    for sp in split:

        if sp == "train":
            part1 = get_data("../data/fl/train-00000-of-00006.parquet")
            part2 = get_data("../data/fl/train-00001-of-00006.parquet")
            part3 = get_data("../data/fl/train-00002-of-00006.parquet")
            part4 = get_data("../data/fl/train-00003-of-00006.parquet")
            part5 = get_data("../data/fl/train-00004-of-00006.parquet")
            part6 = get_data("../data/fl/train-00005-of-00006.parquet")
            # 使用concat追加part2到part1
            final_part = pd.concat([part1, part2, part3, part4, part5, part6], ignore_index=True)
        elif sp == "validation":
            part1 = get_data("../data/fl/{}-00000-of-00002.parquet".format(sp))
            part2 = get_data("../data/fl/{}-00001-of-00002.parquet".format(sp))
            final_part = pd.concat([part1, part2], ignore_index=True)
        else:
            part1 = get_data("../data/fl/{}-00000-of-00002.parquet".format(sp))
            part2 = get_data("../data/fl/{}-00001-of-00002.parquet".format(sp))
            final_part = pd.concat([part1, part2], ignore_index=True)

        caption_data = json.load(open("data/llava_caption_{}.json".format(sp)))

        process_data(final_part, caption_data, sp)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # caption_generate()
    caption_merge()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
