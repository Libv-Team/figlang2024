# This is a sample Python script.
import os
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

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

model_base = "../models/llava-1.5-7b-hf"
model_path = "./checkpoints/llava-1.5-7b-hf-task-lora"

model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)


def get_response(image, prompt):
    prompt = "USER: <image>\n{}\nASSISTANT:\n".format(prompt)

    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    # print(processor.decode(output[0][:], skip_special_tokens=True))
    return processor.decode(output[0][:], skip_special_tokens=True)


# "Give you 2 examples: \n" \
# "1. `Contradiction. The image shows earth covered with snow, with a silhouette of a baby covered in a warm blanket evoking the warmth and care of a mother's embrace, which is the opposite of feeling exposed and vulnerable.`\n" \
# "2. `Entailment. The image displays RoboCop, a character from a science fiction film who is a police officer brought back to life as a cyborg after his death. The text above reads: \"DUDE DIED, BUT THEY MADE HIM GO TO WORK ANYWAY.\" This entails the claim that even death won't exempt you from going to work because it humorously illustrates a character who has been reanimated as a cyborg to continue working despite having died.`\n" \
def train(df):
    file_name = "gemini"
    result = checkpoint_utils.load_checkpoint("result/2/_{}.json".format(file_name))
    skip_rows = len(result)
    print("跳过：{}".format(skip_rows))
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if index < skip_rows:
            continue  # 跳过剩余的代码，继续下一个迭代
        image_bytes = row["image"]
        claim = row["claim"]
        # label = row["label"]
        # explanation = row["explanation"]

        prompt1 = "The claim of this picture is {}. " \
                  "You need to predict this picture is 'entailment' or 'contradiction' first, only use this two word. " \
                  "Then you need to give a explanation for the prediction. " \
                  "The prediction and the explanation are related to the meaning of the figurative language " \
                  "expression. " \
                  "Your response must follow the format shown as below: \n" \
                  "Prediction | Explanation.`\n".format(claim)
        # 使用 BytesIO 将字节数据转换为文件类对象
        image_stream = io.BytesIO(image_bytes["bytes"])

        # 使用 Image.open 读取这个对象
        image = Image.open(image_stream)

        explanation = get_response(image, prompt1)

        label = 1 if "entail" in " ".join(explanation.lower().split("\n")[5:]) else 0
        explanation_final = explanation.split("\n")[-1].replace("Explanation:", "").strip()

        result.append({
            "id": index,
            "label": label,
            "explanation": explanation_final,
        })

        def save():
            checkpoint_utils.save_checkpoint(result,
                                             "result/2/_{}.json".format(file_name))

        my_thread = threading.Thread(target=save)
        my_thread.start()

    r = pd.DataFrame(result)

    # 将DataFrame保存为CSV文件
    csv_filename = "result/2/{}.csv".format(file_name)
    r.to_csv(csv_filename, index=False)


def get_data(path):
    df = pd.read_parquet(path)
    return df


def main():
    part1 = get_data("../data/fl/test-00000-of-00002.parquet")
    part2 = get_data("../data/fl/test-00001-of-00002.parquet")
    # final_part = part1 + part2
    # 使用concat追加part2到part1
    final_part = pd.concat([part1, part2], ignore_index=True)

    train(final_part)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
