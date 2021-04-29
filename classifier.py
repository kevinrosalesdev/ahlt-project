import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def read_features(filename):
    with open(filename) as f:
        feats = []
        labels = []
        infos = []
        for line in f.readlines():
            if line == '\n':
                break
            splitted_line = line.split()
            infos.append(splitted_line[0:3])
            labels.append(splitted_line[3])
            feats.append(splitted_line[4:])
    return feats, labels, infos


if __name__ == '__main__':
    model_name = sys.argv[1]
    test_features = sys.argv[2]
    mlb_name = sys.argv[3]

    x_test, _, train_infos = read_features(test_features)

    f = open(mlb_name, 'rb')
    mlb = pickle.load(f)    # type: MultiLabelBinarizer
    f.close()

    sparse_test_x = mlb.transform(x_test)

    f = open(model_name, 'rb')
    model = pickle.load(f)  # type: RandomForestClassifier
    f.close()

    predictions = model.predict(sparse_test_x)

    # if the classifier predicted a DDI, output it in the right format
    for idx, prediction in enumerate(predictions):
        if prediction != 'null':
            print(train_infos[idx][0], train_infos[idx][1], train_infos[idx][2], prediction, sep='|')
