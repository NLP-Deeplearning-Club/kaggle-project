import pandas as pd
legal_labels = '''bed bird cat dog down eight five four go \
happy house left marvin nine no off on one right seven sheila \
six stop three tree two up wow yes zero'''.split()


def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))
