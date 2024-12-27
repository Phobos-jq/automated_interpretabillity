"""
To get the max activations
"""
import os
import signal
from tqdm import tqdm
import numpy as np
import tiktoken
import datasets as d  # huggingface datasets
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import gc
import json

max_length = 64
batch_size = 2048
MEAN_ACT = False # True: get the most activated sentence(measured by mean activations)
save_file_path = '/data/jqliu/ML_jq/nanoGPT/activations/'
folder_name = 'ori_finetune_nonneg_feature_176000it'
save_file_name = 'neurons_activations'  # '.../folder_name/neurons_activations__len_{max_length}__shard_{shard_index}.json'

model_dir = '/data/jqliu/ML_jq/nanoGPT/out_ori_finetune_nonneg_feature/out_test' # ignored if init_from is not 'resume'
enc = tiktoken.get_encoding("gpt2")

folder_path = os.path.join(save_file_path, folder_name)
if not os.path.exists(folder_path):
    print("Making", folder_path)
    os.mkdir(folder_path)
print("Now geting activations in neurons")
print(f"{max_length=}, {folder_path=}, {model_dir=}, {MEAN_ACT=}")

# 定义信号处理函数
def signal_handler(sig, frame):
    print("捕获到中断信号，正在清理资源...")
    # 在此添加任何清理代码，例如关闭文件、释放资源等
    if 'executor' in globals():
        executor.shutdown(wait=True)
    # 如果有必要，执行其他清理操作
    gc.collect()
    print("资源清理完毕，程序退出。")
    exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# 假设 activations 是激活值字典
def convert_activations_to_json(activations):
    json_ready_activations = {}

    for layer, neurons in activations.items():
        json_ready_activations[layer] = {}
        for neuron, data in neurons.items():
            # 将 tensor 和 numpy 数据转换为可序列化的 Python 类型
            json_ready_activations[layer][neuron] = {
                "max_activation_value": (
                    data["max_activation_value"].item() 
                    if isinstance(data["max_activation_value"], (torch.Tensor, np.generic)) 
                    else data["max_activation_value"]
                ),
                "max_token_idx": (
                    int(data["max_token_idx"].item()) 
                    if isinstance(data["max_token_idx"], (torch.Tensor, np.generic)) 
                    else int(data["max_token_idx"])
                ),
                "max_example_index": int(data["max_example_index"]),
                "max_token_index": int(data["max_token_index"]),
                "activations": data["activations"].tolist(),
                "mean_activation": (
                    data["mean_activation"].item() 
                    if isinstance(data["mean_activation"], (torch.Tensor, np.generic)) 
                    else data["mean_activation"]
                ),
                "activations_first_sample": data["activations_first_sample"].tolist()
            }
    return json_ready_activations

def create_batches(tokenized_shard, batch_size=32):
    """将tokenized的数据集分成指定大小的批次."""
    for i in range(0, len(tokenized_shard), batch_size):
        yield tokenized_shard[i:i + batch_size]

def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out
# tokenize the dataset

def update_max_activations(activation_cache, activations, batch_ids, batch_index, layers_to_hook):
    """更新每个神经元的最大激活值."""
    # 将所有层的激活拼接到一起，形状为 (num_layers, batch_size, seq_len, num_neurons)
    layers_activations = torch.stack([activation_cache[layer_name] for layer_name in layers_to_hook], dim=0)
    mean_activations = layers_activations.mean(dim=2)  # 形状: (num_layers, batch_size, num_neurons)
    num_layers, batch_size, seq_len, num_neurons = layers_activations.shape
    
    if MEAN_ACT == False:
        # 在 batch 和 seq_len 维度上找到每个神经元的最大激活值，保持在 GPU 上
        max_values_in_batch, max_positions = layers_activations.view(num_layers, batch_size * seq_len, num_neurons).max(dim=1)
        
        # 将索引转换为 (batch, token) 索引
        example_indices, token_indices = torch.div(max_positions, seq_len, rounding_mode='floor'), max_positions % seq_len

        # 更新每个神经元的最大激活值
        for layer_idx in range(num_layers):
            for neuron_idx in range(num_neurons):
                max_value = max_values_in_batch[layer_idx, neuron_idx]
                current_max_activation = activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_activation_value"]
                
                if max_value > current_max_activation:
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_activation_value"] = max_value.item()
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_token_idx"] = batch_ids[example_indices[layer_idx, neuron_idx]][token_indices[layer_idx, neuron_idx]]
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_example_index"] = (batch_index * batch_size + example_indices[layer_idx, neuron_idx])
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_token_index"] = token_indices[layer_idx, neuron_idx]
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["activations"] = layers_activations[layer_idx, example_indices[layer_idx, neuron_idx], :, neuron_idx].detach().clone()
    elif MEAN_ACT == True:
        max_mean_act_in_batch, max_positions = mean_activations.max(dim=1)
        for layer_idx in range(num_layers):
            for neuron_idx in range(num_neurons):
                if max_mean_act_in_batch[layer_idx][neuron_idx] > activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["mean_activation"]:
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_example_index"] = batch_index * batch_size + max_positions[layer_idx, neuron_idx]
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["activations"] = layers_activations[layer_idx, max_positions[layer_idx, neuron_idx], :, neuron_idx].detach().clone()
                    activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["mean_activation"] = max_mean_act_in_batch[layer_idx, neuron_idx].item()


    
    # for layer_idx, layer_name in enumerate(layers_to_hook):
    #     activations_layer = activation_cache[layer_name] # 形状: (batch_size, seq_len, num_neurons)
    #     # print("!!", activations_layer.shape)
    #     batch_size, seq_len, num_neurons = activations_layer.shape

    #     # 在 batch 维度上找到每个神经元的最大激活值 (形状: (num_neurons,))
    #     max_values_in_batch, max_positions = activations_layer.view(batch_size * seq_len, num_neurons).max(dim=0)

    #     # 将索引转换为 (batch, token) 索引
    #     example_indices, token_indices = np.divmod(max_positions.cpu().numpy(), seq_len)

    #     for neuron_idx in range(num_neurons):
    #         max_value = max_values_in_batch[neuron_idx]
    #         if max_value > activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_activation_value"]:
    #             activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_activation_value"] = max_value.item()
    #             activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_token_idx"] = batch_ids[example_indices[neuron_idx]][token_indices[neuron_idx]]
    #             activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_example_index"] = batch_index * batch_size + example_indices[neuron_idx]
    #             activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["max_token_index"] = token_indices[neuron_idx]
    #             activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["activations"] = activations_layer[example_indices[neuron_idx], :, neuron_idx].detach().clone()
    
    # 将第一个example作为随机取样，记录activations
    if batch_index == 0:
        for layer_idx, layer_name in enumerate(layers_to_hook):
            for neuron_idx in range(num_neurons):
                activations[f"layer_{layer_idx}"][f"neuron_{neuron_idx}"]["activations_first_sample"] = layers_activations[layer_idx, 0, :, neuron_idx].detach().clone()


def process_shard(shard_index, dataset_sharded, start_event):
    start_event.wait()  # 等待主进程的信号
    try:
        init_from = 'resume' # either 'resume' (from an model_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
        seed = 1337
        device_gpt = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        compile = True # use PyTorch 2.0 to compile the model to be faster
        exec(open('configurator.py').read()) # overrides from command line or config file
        # -----------------------------------------------------------------------------

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in device_gpt else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # model
        if init_from == 'resume':
            # init from a model saved in a specific directory
            ckpt_path = os.path.join(model_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=device_gpt)
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif init_from.startswith('gpt2'):
            # init from a given GPT-2 model
            model = GPT.from_pretrained(init_from, dict(dropout=0.0))

        model.eval()
        model.to(device_gpt)
        if compile:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)

        # look for the meta pickle in case it is available in the dataset folder
        load_meta = False
        if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
            meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
            load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            # ok let's assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)

        # =======模型加载完成=======
        # -----------------------------------------------------------------------------
        tokenized_shard_part = dataset_sharded[f"train_shard_{shard_index}"].map(
            process,
            remove_columns=['text'],
            desc=f"tokenizing shard {shard_index}",
            num_proc=num_proc,
        )
        print(f"shard_{shard_index} is being processed")

        layers_to_hook = [f"transformer.h.{layer}.mlp.gelu" for layer in range(12)]  # 捕捉的层

        # 用于存储每个神经元的激活值情况
        activations = {}
        for i in range(12):
            activations[f"layer_{i}"] = {}
            for j in range(3072):
                activations[f"layer_{i}"][f"neuron_{j}"] = {
                    "max_activation_value": torch.tensor(0.0).to(device_gpt),
                    "max_token_idx": torch.tensor(0).to(device_gpt),
                    "max_example_index": torch.tensor(0).to(device_gpt),
                    "max_token_index": torch.tensor(0).to(device_gpt),
                    "activations": torch.zeros(0).to(device_gpt),
                    "mean_activation": torch.tensor(-100.0).to(device_gpt),
                    "activations_first_sample": torch.zeros(0).to(device_gpt)
                }

        # 逐个batch处理
        for batch_idx, batch in enumerate(create_batches(tokenized_shard_part, batch_size=batch_size)):
        
            x = [torch.tensor(batch['ids'][i][:max_length]).to(device_gpt) for i in range(len(batch['ids']))]
            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=50256).to(device_gpt)

            # 捕捉激活值
            logits, _, activation_cache = model.run_with_cache(x, layers_to_hook=layers_to_hook, device=device_gpt)
            # print("!!start update")
            # 更新最大激活值
            update_max_activations(activation_cache, activations, x, batch_idx, layers_to_hook)

            print(f"shard_{shard_index} batch_{batch_idx} finished")
            if batch_idx % 50 == 0:
                print(activations[f"layer_11"][f"neuron_0"])
                print(activations[f"layer_11"][f"neuron_1"])
            # if shard_index <= 9 and batch_idx % 180 == 0:
            #     # 将 activations 转换为可序列化的字典
            #     json_ready_activations = convert_activations_to_json(activations)
            #     # 将其保存为 JSON 文件
            #     with open(f'/data/jqliu/ML_jq/nanoGPT/activations/activations_shard_{shard_index}_batch_{batch_idx}.json', 'w') as json_file:
            #         json.dump(json_ready_activations, json_file, cls=NpEncoder)
            #     print(f"Shard {shard_index} batch {batch_idx} activations is saved")

        json_ready_activations = convert_activations_to_json(activations)
        # 将其保存为 JSON 文件
        file_name = save_file_name + f'__len_{max_length}__shard_{shard_index}.json'
        save_file = os.path.join(folder_path, file_name)
        with open(save_file, 'w') as json_file:
            json.dump(json_ready_activations, json_file, cls=NpEncoder)
        print(f"Shard {shard_index} batch {batch_idx} activations is saved")
    finally:
        # 清理GPU资源和内存
        del model  # 删除模型以释放显存
        del activation_cache  # 删除激活缓存以释放显存
        torch.cuda.empty_cache()  # 清理未使用的显存
        gc.collect()  # 强制进行垃圾回收

# load the dataset
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

dataset_sharded = d.DatasetDict()

if __name__ == '__main__':
    # 创建事件对象
    manager = mp.Manager()
    start_event = manager.Event()
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = d.load_dataset("/data2/datasets/openwebtext", num_proc=num_proc_load_dataset)
    """
    print(dataset)形如：
    
    DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 8013769
        })
    })
    """
   
    # Now we can shard the tokenized dataset into 64 parts
    num_shards = 64
    
    # Create 64 shards for each dataset split
    for split, dset in dataset.items():
        # 注意初始时split只有一个，即train
        sharded = dset.shard(num_shards=num_shards, index=0, contiguous=True).with_format('numpy')
        dataset_sharded[split] = sharded

        for shard_idx in range(num_shards):
            shard = dset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
            dataset_sharded[f"{split}_shard_{shard_idx}"] = shard


    mp.set_start_method('spawn', force=True)
     # 启动子进程
    with ProcessPoolExecutor(max_workers=2) as executor:
        try:
            futures = [executor.submit(process_shard, shard_idx, dataset_sharded, start_event) for shard_idx in range(num_shards)]
            # 初始化完成后设置事件
            start_event.set()

            # 等待所有子进程完成
            for future in futures:
                try:
                    future.result()  # 确保子进程完成，并抛出异常
                    gc.collect()  # 清理内存
                    torch.cuda.empty_cache()  # 清理GPU缓存
                except Exception as e:
                    print(f"Error in future: {e}")
        except Exception as e:
            print(f"错误：{e}")

        # Force garbage collection
        gc.collect()
        print("All shards processed")
       
