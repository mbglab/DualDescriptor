import math
import random
from statistics import correlation, mean
from nDDpm import NumDualDescriptorPM  # 导入提供的类

class LayerNorm:
    """Layer Normalization implementation (层归一化实现)"""
    def __init__(self, normalized_shape, eps=1e-5):
        """
        初始化层归一化
        Args:
            normalized_shape: 归一化维度大小
            eps: 数值稳定性常数
        """
        self.eps = eps
        self.gamma = [1.0] * normalized_shape  # 缩放参数
        self.beta = [0.0] * normalized_shape   # 平移参数
    
    def __call__(self, x):
        """
        应用层归一化到输入向量
        Args:
            x: 输入向量
        Returns:
            归一化后的向量
        """
        # 计算均值和方差
        mean_val = sum(x) / len(x)
        var_val = sum((xi - mean_val) ** 2 for xi in x) / len(x)
        std_val = math.sqrt(var_val + self.eps)
        
        # 归一化和缩放
        normalized = [(xi - mean_val) / std_val for xi in x]
        return [self.gamma[i] * normalized[i] + self.beta[i] for i in range(len(x))]

class DDN:
    """Dual Descriptor Network (对偶描述子网络)"""
    def __init__(self, layers_config):
        """
        初始化DDN网络
        Args:
            layers_config: 每层的配置字典列表
        """
        self.layers = []
        self.norms = []
        
        # 创建每层描述子和对应的归一化层
        for config in layers_config:
            # 确保所有层使用rank=1以保持序列长度
            config['rank'] = 1
            self.layers.append(NumDualDescriptorPM(**config))
            self.norms.append(LayerNorm(config['vec_dim']))
    
    def forward(self, seq):
        """
        前向传播：处理序列通过所有层
        Args:
            seq: 输入向量序列
        Returns:
            处理后的序列
        """
        current = seq
        for layer, norm in zip(self.layers, self.norms):
            # 计算当前层描述
            described = layer.describe(current)
            
            # 检查序列长度是否一致
            if len(described) != len(current):
                # 如果长度不一致，使用最后一个有效描述向量填充
                last_valid = described[-1] if described else [0.0] * len(current[0])
                described = described + [last_valid] * (len(current) - len(described))
            
            # 残差连接
            residual = []
            for i in range(len(current)):
                res_vec = [current[i][d] + described[i][d] for d in range(len(current[0]))]
                residual.append(res_vec)
            
            # 层归一化
            current = [norm(vec) for vec in residual]
        
        return current
    
    def predict_t(self, seq):
        """
        预测目标向量（平均所有位置）
        Args:
            seq: 输入向量序列
        Returns:
            预测的目标向量
        """
        output = self.forward(seq)
        if not output:
            return [0.0] * len(seq[0])
        
        # 计算所有输出向量的平均值
        avg_vec = [0.0] * len(output[0])
        for vec in output:
            for d in range(len(vec)):
                avg_vec[d] += vec[d]
        return [x / len(output) for x in avg_vec]
    
    def train_layers(self, seqs, t_list, max_iters=300, learning_rate=0.1, decay_rate=0.99, print_every=20):
        """
        分层监督训练网络（使用grad_train方法）
        
        Args:
            seqs: 训练序列列表
            t_list: 目标向量列表
            max_iters: 最大训练迭代次数
            learning_rate: 初始学习率
            decay_rate: 学习率衰减率
            print_every: 打印进度间隔
        """
        # 训练第一层
        print(f"\nTraining layer 1/{len(self.layers)}")
        self._train_layer(0, seqs, t_list, max_iters, learning_rate, decay_rate, print_every)
        
        # 训练后续层（使用前一层的输出作为输入）
        for i in range(1, len(self.layers)):
            print(f"\nTraining layer {i+1}/{len(self.layers)}")
            # 获取前一层的输出序列
            transformed_seqs = [self.forward(seq) for seq in seqs]
            self._train_layer(i, transformed_seqs, t_list, max_iters, learning_rate, decay_rate, print_every)
    
    def _train_layer(self, layer_idx, seqs, t_list, max_iters, learning_rate, decay_rate, print_every):
        """
        训练单个层
        Args:
            layer_idx: 层索引
            seqs: 输入序列列表
            t_list: 目标向量列表
            max_iters: 最大迭代次数
            learning_rate: 初始学习率
            decay_rate: 学习率衰减率
            print_every: 打印间隔
        """
        layer = self.layers[layer_idx]
        
        # 使用梯度下降训练当前层
        history = layer.grad_train(
            seqs, 
            t_list, 
            max_iters=max_iters,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            print_every=print_every
        )
        
        # 保存训练历史（可选）
        return history

# 示例用法
if __name__ == "__main__":
    # 配置参数
    vec_dim = 4  # 向量维度
    num_seqs = 30  # 序列数量（增加数据量）
    min_len, max_len = 100, 200  # 序列长度范围

    # 创建合成数据
    #random.seed(42)
    seqs = []
    t_list = []
    print("Generating synthetic data...")
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # 创建具有时间依赖性的序列
        base = [random.uniform(-1, 1) for _ in range(vec_dim)]
        seq = []
        for i in range(L):
            # 添加时间依赖的噪声
            vec = [base[d] + 0.1 * math.sin(i * 0.1 + d * 0.5) for d in range(vec_dim)]
            seq.append(vec)
        seqs.append(seq)
        
        # 创建与序列内容相关的目标向量
        t_vec = [sum(vec[d] for vec in seq) / L for d in range(vec_dim)]
        t_list.append(t_vec)    
    
    # 配置DDN网络 (3层)
    ddn_config = [
        {'vec_dim': vec_dim, 'num_basis': 7, 'mode': 'linear'},
        {'vec_dim': vec_dim, 'num_basis': 5, 'mode': 'linear'},
        {'vec_dim': vec_dim, 'num_basis': 3, 'mode': 'linear'}
    ]
    ddn = DDN(ddn_config)
    
    # 训练网络（使用梯度下降监督训练）
    print("\nStarting DDN training with grad_train...")
    ddn.train_layers(
        seqs, 
        t_list,
        max_iters=300,
        learning_rate=0.2,
        decay_rate=0.9999,
        print_every=30
    )
    
    # 评估网络性能
    print("\nEvaluating DDN performance...")
    # 使用整个网络进行预测
    preds = [ddn.predict_t(seq) for seq in seqs]
    
    # 计算相关性
    correlations = []
    for d in range(vec_dim):
        actual = [t[d] for t in t_list]
        predicted = [p[d] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {d} correlation: {corr:.4f}")
    
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # 重构示例序列
    print("\nReconstructing sequence with DDN...")
    reconstructed = ddn.forward(seqs[0])
    print(f"Original first vector: {[round(x, 3) for x in seqs[0][0]]}")
    print(f"Reconstructed first vector: {[round(x, 3) for x in reconstructed[0]]}")
    
    # 特征提取
    print("\nExtracting DDN features...")
    feature_vec = []
    for vec in reconstructed:
        feature_vec.extend(vec)
    print(f"Feature vector length: {len(feature_vec)}")
    print(f"First 5 features: {[round(x, 4) for x in feature_vec[:5]]}")

    # 单层与多层网络性能比较
    print("\nComparing single layer vs DDN performance:")
    # 创建单层模型（与第一层相同配置）
    single_layer = NumDualDescriptorPM(vec_dim=vec_dim, rank=1, num_basis=7)
    # 使用相同的grad_train方法训练
    single_layer.grad_train(
        seqs, 
        t_list,
        max_iters=300,
        learning_rate=0.2,
        decay_rate=0.9999,
        print_every=30
    )
    
    single_preds = [single_layer.predict_t(seq) for seq in seqs]
    single_corrs = []
    for d in range(vec_dim):
        actual = [t[d] for t in t_list]
        predicted = [p[d] for p in single_preds]
        corr = correlation(actual, predicted)
        single_corrs.append(corr)
    
    print(f"Single layer avg correlation: {mean(single_corrs):.4f}")
    print(f"DDN avg correlation: {mean(correlations):.4f}")
    print(f"Improvement: {(mean(correlations) - mean(single_corrs)):.4f}")

    # 保存和加载模型演示
    print("\nSaving and loading model...")
    ddn.layers[0].save("ddn_layer1.pkl")
    loaded_layer = NumDualDescriptorPM.load("ddn_layer1.pkl")
    
    # 验证加载的模型
    test_seq = seqs[0]
    pred_original = ddn.layers[0].predict_t(test_seq)
    pred_loaded = loaded_layer.predict_t(test_seq)
    
    print(f"Original layer prediction: {[round(x, 4) for x in pred_original[:2]]}...")
    print(f"Loaded layer prediction: {[round(x, 4) for x in pred_loaded[:2]]}...")
    
    # 计算预测差异
    diff = sum((a - b) ** 2 for a, b in zip(pred_original, pred_loaded)) ** 0.5
    print(f"Prediction difference: {diff:.6f}")
