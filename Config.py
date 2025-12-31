import torch


class Config:
    num_classes = 10
    cond_emb_dim = 128
    timesteps = 500  # 扩散时间步数
    batch_size = 5  # 批大小
    lr = 2e-4  # 学习率
    epochs = 2000  # 训练轮数
    dropout_prob = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 配置参数
    DATA_DIR = "./data/preprocess_bearing/0HP"
    SEGMENT_LENGTH = 1024  # 信号段长度
    DF_TRAIN_SAMPLES_PER_CLASS = 30  # 每类训练样本数
    TRAIN_SAMPLES_PER_CLASS = 1000  # 每类训练样本数
    DF_TEST_SAMPLES_PER_CLASS = 0  # 每类测试样本数
    TEST_SAMPLES_PER_CLASS = 200  # 每类测试样本数
    OVERLAP = True  # 启用样本重叠
    DF_STEP_SIZE = 1024
    STEP_SIZE = 128
    # 数据参数
    dim = 1
    seq_len = 1024
    # 样本标签
    label = 0
    #0 0.96 1 0.98 2 0.97 4 0.92 5 0.92 样本增加到50个就解决了
    #0 97 2 98 4 98 1 98 3 99 5 98 6  7 8 9