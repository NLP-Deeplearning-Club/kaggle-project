import pandas as pd
legal_labels = '''bed bird cat dog down eight five four go \
happy house left marvin nine no off on one right seven sheila \
six stop three tree two up wow yes zero'''.split()


def label_transform(labels):
    """将样本标签数据转化为one-hot编码,也就是说将一个常数转化为一个长度为标签数量+1的向量

    Parameters:
        labels (list): - 标签按特征顺序排列的集合

    Returns:
        np.ndarray: - 训练集(音频路径,标签)和测试集(音频路径,标签)组成的元组
    """
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    table = pd.get_dummies(pd.Series(nlabels))
    return table
