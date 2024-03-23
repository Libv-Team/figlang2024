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

model_finetune = "./finetune/llava-v1.5-7b-lora-oc1"
num = "oc1-256"
# model_path = "./checkpoints/llava-1.5-7b-hf-task-lora"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_finetune,
#     model_base=None,
#     model_name=get_model_name_from_path(model_finetune)
# )

# model_path = "bczhou/TinyLLaVA-3.1B"
# prompt = "What are the things I should be cautious about when I visit here?"
# image_file = "https://llava-vl.github.io/static/images/view.jpg"

model_name = get_model_name_from_path(model_finetune)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_finetune, None, model_name
)


def get_response(image_path, qs):
    qs = "USER: {}\nASSISTANT:\n".format(qs.replace("<image>", ""))
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # if 'phi' in model_name.lower() or '3.1b' in model_name.lower():
    #     conv_mode = "phi"
    # if "llama-2" in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"
    #
    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print(
    #         "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
    #             conv_mode, args.conv_mode, args.conv_mode
    #         )
    #     )
    # else:
    # args.conv_mode = conv_mode

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = [image_path]
    images = load_images(image_files)

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    """
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": "phi",
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    """

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=256,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(
    #         f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
    #     )
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs


# "Give you 2 examples: \n" \
# "1. `Contradiction. The image shows earth covered with snow, with a silhouette of a baby covered in a warm blanket evoking the warmth and care of a mother's embrace, which is the opposite of feeling exposed and vulnerable.`\n" \
# "2. `Entailment. The image displays RoboCop, a character from a science fiction film who is a police officer brought back to life as a cyborg after his death. The text above reads: \"DUDE DIED, BUT THEY MADE HIM GO TO WORK ANYWAY.\" This entails the claim that even death won't exempt you from going to work because it humorously illustrates a character who has been reanimated as a cyborg to continue working despite having died.`\n" \
def train(data, image_folder):
    file_name = "gemini"
    result = checkpoint_utils.load_checkpoint("result/{}/_{}.json".format(num, file_name))
    skip_rows = len(result)
    print("跳过：{}".format(skip_rows))
    for index, row in tqdm(enumerate(data), total=len(data)):
        if index < skip_rows:
            continue  # 跳过剩余的代码，继续下一个迭代
        image_path = image_folder + row["image"]
        prompt = row["conversations"][0]["value"]

        explanation = get_response(image_path, prompt)

        label = 1 if "entail" in explanation else 0
        explanation_final = ".".join(explanation.split(".")[1:]).strip()

        result.append({
            "id": index,
            "label": label,
            "explanation": explanation_final,
        })

        def save():
            checkpoint_utils.save_checkpoint(result,
                                             "result/{}/_{}.json".format(num, file_name))

        my_thread = threading.Thread(target=save)
        my_thread.start()

    r = pd.DataFrame(result)

    # 将DataFrame保存为CSV文件
    csv_filename = "result/{}/{}.csv".format(num, file_name)
    r.to_csv(csv_filename, index=False)


def get_data(path):
    df = pd.read_parquet(path)
    return df


def main():
    file = "./data/test_ac2.json"
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
