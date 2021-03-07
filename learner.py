import sys
import pycrfsuite


def read_features(filename):
    x = []
    y = []
    infos = []
    sids = []
    with open(filename) as f:
        feats = []
        labels = []
        info = []
        for line in f.readlines():
            if line == '\n':
                x.append(feats)
                y.append(labels)
                infos.append(info)
                sids.append(splitted_line[0])
                feats = []
                labels = []
                info = []
                continue
            splitted_line = line.split()
            labels.append(splitted_line[4])
            feats.append(splitted_line[5:])
            info.append(splitted_line[1:4])
    return x, y, sids, infos


if __name__ == '__main__':
    model_name = sys.argv[1]
    features = sys.argv[2]
    x_train, y_train, _, _ = read_features(features)
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(f'{model_name}.crfsuite')
