import sys

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC


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
    train_features = sys.argv[2]
    mlb_name = sys.argv[3]

    x_train, y_train, train_infos = read_features(train_features)
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_train_x = mlb.fit_transform(x_train)

    f = open(mlb_name, 'wb')
    pickle.dump(mlb, f)
    f.close()

    rfc = RandomForestClassifier(n_estimators=120, criterion='entropy', random_state=0)
    rfc.fit(sparse_train_x, y_train)

    f = open(model_name, 'wb')
    pickle.dump(rfc, f)
    f.close()
