import json
import os
import math
from activations import ActivationRecord,NeuronId,NeuronRecord

max_length = 64
activations_dir = '/data/jqliu/ML_jq/nanoGPT/activations/'
folder_name = 'ori_finetune_nonneg_feature_250000it'
neuron_records_name = os.path.join(activations_dir, folder_name, 'neuron_records') # +_neurons.json/_features.json

import pickle
from contextlib import nullcontext
import torch
import tiktoken
import datasets as d
from model import GPTConfig, GPT

info_num = 50
shard_num = info_num
num_proc = 8
num_proc_load_dataset = num_proc
num_shards = 64
max_length = 64
enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)

def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out        

def create_from_info_dict_features():
    info_file_name = 'info_dict_features_activations'
    info_dict_path = os.path.join(activations_dir, folder_name, info_file_name+f'__len_{max_length}.json')
    neuron_records_path = neuron_records_name + '_featuress.json'
    print("create_from_info_dict_features")
    print(f'{info_dict_path=}')
    print(f'{neuron_records_path=}')
    with open(info_dict_path, 'r') as f:
        info_dict = json.load(f)
    
    neuron_records = {}
    for i in range(1):
        neuron_records[f"layer_{i}"] = {}
        for j in range(768):
            neuron_records[f"layer_{i}"][f"neuron_{j}"] = {}

    for layer in range(1):
        # layers_to_hook = ["features"]  # 可以根据需要调整捕捉的层
        for neuron in range(768):
            neuron_info = info_dict[f"feature_{neuron}"]

            # 构建 ActivationRecord 列表
            random_sample = []
            most_positive_activation_records = []
            
            for i in range(neuron_info["num_info"]):
                activation_record_max_sample = ActivationRecord(
                    tokens=neuron_info["max_sequence"][i],  # 提取token
                    activations=neuron_info["activations"][i]  # 提取激活值
                )
                activation_record_random_sample = ActivationRecord(
                    tokens=neuron_info["first_sample_sequence"][i],  # 提取token
                    activations=neuron_info["activations_first_sample"][i]  # 提取激活值
                )
                most_positive_activation_records.append(activation_record_max_sample)
                random_sample.append(activation_record_random_sample)
            
            # 获取激活的统计数据
            mean = neuron_info.get("mean", math.nan)  # 默认值为nan
            variance = neuron_info.get("variance", math.nan)
            skewness = neuron_info.get("skewness", math.nan)
            kurtosis = neuron_info.get("kurtosis", math.nan)

            # 构建 NeuronRecord
            neuron_record = NeuronRecord(
                neuron_id=NeuronId(layer_index=layer, neuron_index=neuron),
                random_sample=random_sample,
                most_positive_activation_records=most_positive_activation_records,  # 加入激活记录
                mean=mean,  # 统计信息
                variance=variance,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
            neuron_records[f"layer_{layer}"][f"neuron_{neuron}"] = neuron_record
            if neuron%300 == 0:
                print(f"layer_{layer}, neuron_{neuron} finished")

    with open(neuron_records_path, 'w') as f:
        json.dump({layer: {neuron: record.to_dict() for neuron, record in neurons.items()}
                for layer, neurons in neuron_records.items()}, f)
    # with open(neuron_records_path, 'w') as f:
    #     json.dump(neuron_records, f)
    print("finish")

def create_from_info_dict_neurons():
    info_file_name = 'info_dict_neurons_activations'
    info_dict_path = os.path.join(activations_dir, folder_name, info_file_name+f'__len_{max_length}.json')
    neuron_records_path = neuron_records_name + '_neurons.json'
    print("create_from_info_dict_neurons")
    print(f'{info_dict_path=}')
    print(f'{neuron_records_path=}')
    with open(info_dict_path, 'r') as f:
        info_dict = json.load(f)

    neuron_records = {}
    for i in range(12):
        neuron_records[f"layer_{i}"] = {}
        for j in range(3072):
            neuron_records[f"layer_{i}"][f"neuron_{j}"] = {}

    for layer in range(12):
        # layers_to_hook = [f"transformer.h.{layer}.mlp.gelu"]  # 可以根据需要调整捕捉的层
        for neuron in range(3072):
            neuron_info = info_dict[f"layer_{layer}"][f"neuron_{neuron}"]

            # 构建 ActivationRecord 列表
            random_sample = []
            most_positive_activation_records = []
            
            for i in range(neuron_info["num_info"]):
                activation_record_max_sample = ActivationRecord(
                    tokens=neuron_info["max_sequence"][i],  # 提取token
                    activations=neuron_info["activations"][i]  # 提取激活值
                )
                activation_record_random_sample = ActivationRecord(
                    tokens=neuron_info["first_sample_sequence"][i],  # 提取token
                    activations=neuron_info["activations_first_sample"][i]  # 提取激活值
                )
                most_positive_activation_records.append(activation_record_max_sample)
                random_sample.append(activation_record_random_sample)
            
            # 获取激活的统计数据
            mean = neuron_info.get("mean", math.nan)  # 默认值为nan
            variance = neuron_info.get("variance", math.nan)
            skewness = neuron_info.get("skewness", math.nan)
            kurtosis = neuron_info.get("kurtosis", math.nan)

            # 构建 NeuronRecord
            neuron_record = NeuronRecord(
                neuron_id=NeuronId(layer_index=layer, neuron_index=neuron),
                random_sample=random_sample,
                most_positive_activation_records=most_positive_activation_records,  # 加入激活记录
                mean=mean,  # 统计信息
                variance=variance,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
            neuron_records[f"layer_{layer}"][f"neuron_{neuron}"] = neuron_record
            if neuron%300 == 0:
                print(f"layer_{layer}, neuron_{neuron} finished")

    with open(neuron_records_path, 'w') as f:
        json.dump({layer: {neuron: record.to_dict() for neuron, record in neurons.items()}
                for layer, neurons in neuron_records.items()}, f)
    # with open(neuron_records_path, 'w') as f:
    #     json.dump(neuron_records, f)
    print("finish")
    
if __name__ == '__main__':
    create_from_info_dict_features()


