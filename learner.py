import sys

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier


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


    # parameters = {
    #     #'objective':['multi:softmax','multi:softprob'],
    #     'learning_rate': [0.3],
    #     'max_depth': [4,5,6],
    #     'min_child_weight': [1,3,5],
    #     'subsample': [1,0.75,0.5],
    #     'colsample_bytree': [1,0.75,0.5],
    #     'n_estimators': [300],
    #     'reg_lambda': [1,2,3],
    #     'gamma' : [0,1,2]
    # }
    #
    # rfc = XGBClassifier()
    # gridsearch = GridSearchCV(rfc,verbose=4,scoring='f1_macro',param_grid=parameters,n_jobs=-1)
    # gridsearch.fit(sparse_train_x, y_train)
    # rfc=gridsearch.best_estimator_
    # print(gridsearch.best_params_)
    # rfc.fit(sparse_train_x, y_train)

    parameters ={'colsample_bytree': 0.75,
                 'gamma': 0,
                 'learning_rate': 0.3,
                 'max_depth': 6,
                 'min_child_weight': 1,
                 'n_estimators': 1000,
                 'reg_lambda': 2,
                 'subsample': 1}
    rfc = XGBClassifier(n_jobs=-1,**parameters)
    rfc.fit(sparse_train_x, y_train)

    f = open(model_name, 'wb')
    pickle.dump(rfc, f)
    f.close()
