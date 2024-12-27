"""
To get the max activations info from the saved .json file
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
import datasets as d  # huggingface datasets
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import json


max_length = 64
activations_dir = '/data/jqliu/ML_jq/nanoGPT/activations/'
folder_name = 'nonneg_feature_136000it'        # 调整这个
info_file_name = 'info_dict_neurons_activations'
info_dict_path = os.path.join(activations_dir, folder_name, info_file_name+f'__len_{max_length}.json')
activations_file_name = os.path.join(activations_dir, folder_name, f'neurons_activations__len_{max_length}')

shard_save_num = 50
num_shards = 64
num_proc = 8
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)
print("Now processing activations of neurons")
print(f"{max_length=}, {info_dict_path=}")

def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out
# tokenize the dataset

dataset = d.load_dataset("/data2/datasets/openwebtext", num_proc=num_proc_load_dataset)
# Create 64 shards for each dataset split
dataset_sharded = d.DatasetDict()
for split, dset in dataset.items():
    # 注意初始时split只有一个，即train
    sharded = dset.shard(num_shards=num_shards, index=0, contiguous=True).with_format('numpy')
    dataset_sharded[split] = sharded

    for shard_idx in range(num_shards):
        shard = dset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        dataset_sharded[f"{split}_shard_{shard_idx}"] = shard



info_dict = {}
for i in range(12):
    info_dict[f"layer_{i}"] = {}
    for j in range(3072):
        info_dict[f"layer_{i}"][f"neuron_{j}"] = {
            "max_activation_value": [],
            "mean_activation_value": [],

            "max_token": [],
            "max_token_idx": [],
            "max_sequence": [],             # 一个token列表（注意不是序列，而是已经分词处理过的）
            "max_sequence_tokenized": [],   # 一个整数列表

            "max_token_index": [],
            "activations":[],

            "first_sample_sequence":[],
            "first_sample_sequence_tokenized":[],
            "activations_first_sample":[],
            "num_info": 0
        }


for s_i in range(shard_save_num):
    tokenized_shard_part = dataset_sharded[f"train_shard_{s_i}"].map(
        process,
        remove_columns=['text'],
        desc=f"tokenizing shard {s_i}",
        num_proc=num_proc,
    )
    activations_json = activations_file_name + f"__shard_{s_i}.json"
    # 打开一个文件用于读取
    # 用于存储每个神经元的激活值情况
    activations = {}
    with open(os.path.join(activations_dir, activations_json), 'r') as f:
        # 使用json.load()从文件中读取序列化的对象并还原为原来的Python对象
        activations = json.load(f)

    for i in range(12):
        for j in range(3072):
            sequence_tokenized = tokenized_shard_part[
                activations[f"layer_{i}"][f"neuron_{j}"]["max_example_index"]]['ids'][:max_length]
            first_sample_sequence_tokenized = tokenized_shard_part[0]['ids'][:max_length]
            max_token = decode([activations[f"layer_{i}"][f"neuron_{j}"]["max_token_idx"]])
            if j == 0:
                print(max_token)
            # if sequence_tokenized not in info_dict[f"layer_{i}"][f"neuron_{j}"]["max_sequence_tokenized"]:
            info_dict[f"layer_{i}"][f"neuron_{j}"]["max_activation_value"].append(
                activations[f"layer_{i}"][f"neuron_{j}"]["max_activation_value"])
            info_dict[f"layer_{i}"][f"neuron_{j}"]["mean_activation_value"].append(
                activations[f"layer_{i}"][f"neuron_{j}"]["mean_activation"])
            info_dict[f"layer_{i}"][f"neuron_{j}"]["max_token"].append(
                max_token)
            info_dict[f"layer_{i}"][f"neuron_{j}"]["max_token_idx"].append(
                activations[f"layer_{i}"][f"neuron_{j}"]["max_token_idx"])
            info_dict[f"layer_{i}"][f"neuron_{j}"]["max_sequence_tokenized"].append(
                sequence_tokenized)
            # info_dict[f"layer_{i}"][f"neuron_{j}"]["max_sequence"].append(
            #     dataset_sharded[f"train_shard_{s_i}"][activations[f"layer_{i}"][f"neuron_{j}"]["max_example_index"]]['text']) # 注意这里不能加[:max_length]
            # 上述代码保存的序列太长，长于max_length
            info_dict[f"layer_{i}"][f"neuron_{j}"]["max_sequence"].append(
                [decode([sequence_tokenized[k]]) for k in range(max_length)])

            info_dict[f"layer_{i}"][f"neuron_{j}"]["max_token_index"].append(
                activations[f"layer_{i}"][f"neuron_{j}"]["max_token_index"])
            info_dict[f"layer_{i}"][f"neuron_{j}"]["activations"].append(
                activations[f"layer_{i}"][f"neuron_{j}"]["activations"])
            
            info_dict[f"layer_{i}"][f"neuron_{j}"]["activations_first_sample"].append(
                activations[f"layer_{i}"][f"neuron_{j}"]["activations_first_sample"])
            info_dict[f"layer_{i}"][f"neuron_{j}"]["first_sample_sequence"].append(
                [decode([first_sample_sequence_tokenized[k]]) for k in range(max_length)])
            info_dict[f"layer_{i}"][f"neuron_{j}"]["first_sample_sequence_tokenized"].append(
                first_sample_sequence_tokenized)
            info_dict[f"layer_{i}"][f"neuron_{j}"]["num_info"] += 1

    print(f"shard_{s_i} finish!")

# 保存 info_dict
with open(info_dict_path, 'w') as f:
    json.dump(info_dict, f)

print("finish!")         