import os
import random
import asyncio  # 用于运行异步函数
import logging

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "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"
import sys 
sys.path.append("/home/jqliu/ML_jq/neuronExpainer/automated-interpretability/neuron-explainer")
neuron_records_path = "/data/jqliu/ML_jq/nanoGPT/activations/ori_136000it/neuron_records_neurons.json"

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, load_neuron
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationTokenByTokenSimulator

# 设置日志配置，将所有输出保存到文件
logging.basicConfig(filename='evaluation_ori_136000it_layer10_results.log', level=logging.INFO, format='%(message)s')

EXPLAINER_MODEL_NAME = "gpt-4"
SIMULATOR_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TEST_NUM = 24  # 要抽取测试的神经元数量
SEED = 42  # 设置随机种子，确保可复现性

# 设置随机种子
random.seed(SEED)

async def evaluate_neurons():
    # 创建字典以记录结果
    results = {}
    total_score = 0.0

    # 随机抽取神经元编号
    random_neuron_indices = random.sample(range(3072), TEST_NUM)
    logging.info("=========================")
    logging.info(f"{neuron_records_path=}")
    logging.info(f"{random_neuron_indices=}")
    logging.info("=========================")
    print("=========================")
    print(f"{neuron_records_path=}")
    print(f"{random_neuron_indices=}")
    print("=========================")

    for i, neuron_idx in enumerate(random_neuron_indices, 1):  # 添加索引 i
        logging.info(f"now evaluating feature_{neuron_idx}")
        print(f"now evaluating feature_{neuron_idx}")        
        
        # 加载神经元记录
        neuron_record = load_neuron(10, neuron_idx, neuron_records_path)
        
        # 获取激活记录
        slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
        train_activation_records = neuron_record.train_activation_records(
            activation_record_slice_params=slice_params
        )
        valid_activation_records = neuron_record.valid_activation_records(
            activation_record_slice_params=slice_params
        )

        # 生成神经元解释
        explainer = TokenActivationPairExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
        )
        assert len(explanations) == 1
        explanation = explanations[0]
        logging.info(f"Neuron {neuron_idx} explanation: {explanation}")
        print(f"Neuron {neuron_idx} explanation: {explanation}")

        # 模拟并计算得分
        simulator = UncalibratedNeuronSimulator(
            ExplanationTokenByTokenSimulator(
                SIMULATOR_MODEL_NAME,
                explanation,
                max_concurrent=1,
                prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
            )
        )
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)
        score = scored_simulation.get_preferred_score()
        logging.info(f"Neuron {neuron_idx} score: {score:.2f}")
        print(f"Neuron {neuron_idx} score: {score:.2f}")
        
        # 将结果存入字典
        results[neuron_idx] = {
            "explanation": explanation,
            "score": score
        }
        total_score += score

        # # 每处理 4 个神经元，暂停并询问用户是否继续
        # if i % 4 == 0:
        #     print(results)
        #     user_input = input("Do you want to continue? Type 'YES' to proceed: ")
        #     if user_input.strip().upper() != 'YES':
        #         logging.info("Process interrupted by user.")
        #         print("Process interrupted by user.")
        #         return  # 退出函数

    # 计算平均得分
    average_score = total_score / TEST_NUM
    logging.info("Finished")
    logging.info(f"\nAverage score for {TEST_NUM} neurons: {average_score:.2f}")
    print("Finished")
    print(f"\nAverage score for {TEST_NUM} neurons: {average_score:.2f}")

    # 打印所有结果
    logging.info("\nResults for each neuron:")
    print("\nResults for each neuron:")
    for neuron_idx, data in results.items():
        logging.info(f"Neuron {neuron_idx}: Explanation: {data['explanation']}, Score: {data['score']:.2f}")
        print(f"Neuron {neuron_idx}: Explanation: {data['explanation']}, Score: {data['score']:.2f}")

def main():
    # 运行异步任务
    asyncio.run(evaluate_neurons())

# 检查是否作为主程序运行
if __name__ == "__main__":
    main()
