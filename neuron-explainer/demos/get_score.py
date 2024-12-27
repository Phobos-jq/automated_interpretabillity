import re
import numpy as np

# 文件路径
file_path = 'neuronExpainer/automated-interpretability/neuron-explainer/demos/evaluation_nonneg_feature_136000it_layer10_results.log'

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 使用正则表达式提取所有 score: 后的数字
scores = re.findall(r'score:\s*(-?[\d.]+)', content)
scores = list(map(float, scores))  # 转换为浮点数列表

# 计算均值和方差
mean_score = np.mean(scores)
variance_score = np.var(scores)

# 打印结果
print(f"提取的分数: {scores}")
print(f"样本数: {len(scores)}")
print(f"均值: {mean_score:.4f}")
print(f"方差: {variance_score:.4f}")
