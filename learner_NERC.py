import sys
import pickle
import random
import tensorflow as tf
import numpy as np
import re
from os import listdir
from xml.dom.minidom import parse
from nerc.feature_extractor import tokenize, get_tag
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout, MaxPooling2D,Conv1D
from tensorflow.keras import Model
from tf2crf import CRF, ModelWithCRFLoss


def load_data(path):  # Ruizhe
    x = []
    y = []
    sids = []
    infos = []
    offsets = []
    for idx, f in enumerate(listdir(path), 1):
        # parse XML file, obtaining a DOM tree
        tree = parse(path + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            sids.append(sid)
            stext = s.attributes["text"].value  # get sentence text
            # load ground truth entities
            gold = []
            entities = s.getElementsByTagName("entity")
            for e in entities:
                # for discontinuous entities, we only get the first pan
                offset = e.attributes["charOffset"].value
                (start, end) = offset.split(";")[0].split("-")
                gold.append((int(start), int(end), e.attributes["type"].value))
                infos.append(gold)

            # tokenize text
            tokens = tokenize(stext)
            tags = []
            words = []
            words_offsets = []

            for i in range(0, len(tokens)):
                features = [f'form={tokens[i][0]}']
                # Suffix 5, 4, 3. The word length without suffix must be at least 2+len(suffix)
                if len(tokens[i][0]) >= 7:
                    features.append(f'suf5={tokens[i][0][-5:]}')
                    features.append(f'pref5={tokens[i][0][:5]}')
                if len(tokens[i][0]) >= 6:
                    features.append(f'suf4={tokens[i][0][-4:]}')
                    features.append(f'pref4={tokens[i][0][:4]}')
                if len(tokens[i][0]) >= 5:
                    features.append(f'suf3={tokens[i][0][-3:]}')
                    features.append(f'pref3={tokens[i][0][:3]}')

                # Suffix 2 and 1 (last characters of word)
                if len(tokens[i][0]) >= 4:
                    features.append(f'suf2={tokens[i][0][-2:]}')
                    features.append(f'pref2={tokens[i][0][:2]}')
                features.append(f'suf1={tokens[i][0][-1:]}')
                features.append(f'pref1={tokens[i][0][:1]}')
                # # Capitalized
                # if tokens[i][0].isupper():
                #     features.append('capitalized=True')
                # else:
                #     features.append('capitalized=False')
                #
                # # With numbers
                # if re.findall(r'[0-9]+', tokens[i][0]):
                #     features.append("number=True")
                # else:
                #     features.append("number=False")
                # With dashes
                # if re.findall(r'-+', tokens[i][0]):
                #     features.append("dash=True")
                # else:
                #     features.append("dash=False")

                words.append(features)
                words_offsets.append((tokens[i][1], tokens[i][2]))
                # see if the token is part of an entity, and which part (B/I)
                tag = get_tag(tokens[i], gold)
                tags.append(tag)

            y.append(tags)
            x.append(words)
            offsets.append(words_offsets)


    return x, y, sids, infos, offsets


def create_indexs(x, y, max_length):  # Ruizhe
    feats_aux = {}
    feats = {}

    for f in set(chain(*chain(*x))):
        try:
            name, value = f.split('=')
        except ValueError:
            pass  # for now I ignore true or false features
        try:
            feats_aux[name].add(value)
        except KeyError:
            feats_aux[name] = set(value)

    for name in feats_aux.keys():
        feats[name] = {}
        for i, value in enumerate(feats_aux[name]):
            feats[name][value] = i + 2

    lb = LabelBinarizer()
    lb.fit(list(chain(*y)))

    idx = {'feats': feats,
           'labels': {l: i + 1 for i, l in enumerate(lb.classes_)},
           'max_length': max_length}

    for key in idx['feats'].keys():
        idx['feats'][key]['<PAD>'] = 0
        idx['feats'][key]['<UNK>'] = 1
    idx['labels']['<PAD>'] = 0

    return idx


def build_network(idx):  # Kevin
    """
    Task: Create network for the learner
    Input:
        idx: index dictionary with words/labels codes, plus maximum sentence length
    Output:
        Returns a compiled Keras neural network with the specified layers.
    """

    # sizes
    n_words = len(idx['feats']['form'])
    n_labels = len(idx['labels'])
    n_feats=len(idx['feats'].keys())
    max_len = idx['max_length']
    embedding = 64

    # create network layers
    inp = Input(shape=(n_feats,max_len,))
    # ... add missing layers here ...

    model = Embedding(input_dim=n_words, output_dim=embedding, mask_zero=True)(inp)
    #Conv Model
    # model = Conv1D(128, 5, activation='relu',padding='same')(model)
    # model = MaxPooling2D((n_feats,1))(model)
    # model = Dense(10, activation='relu')(tf.squeeze(model,axis=1))

    #LSTM Model
    model = Bidirectional(LSTM(units=50, return_sequences=True,dropout=0.2, recurrent_dropout=0.1))(tf.reduce_sum(model,axis=1))
    model = TimeDistributed(Dense(50, activation='relu'))(model)
    model = TimeDistributed(Dropout(0.5))(model)
    out = Dense(n_labels, activation='softmax')(model)


    # CRF Layers
    # model = TimeDistributed(Dense(50, activation='relu'))(model)
    # crf = CRF(units=max_len)
    # out = crf(model)


    # create and compile model
    model = Model(inp, out)

    # (Non-CRF compile) set appropriate parameters (optimizer, loss, etc.)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # (CRF compile)
    # model = ModelWithCRFLoss(model, sparse_target=False)
    # model.compile(optimizer='adam')
    return model


def encode_words(x, idx):  # Ruizhe
    max_length = idx['max_length']
    feats = idx['feats']
    X = {key: [] for key in feats.keys()}
    for sentence in x:
        if len(sentence) <= max_length:
            s = sentence + [['<PAD>']] * (max_length - len(sentence))
        if len(sentence) > max_length:
            s = sentence[:max_length]
        inputs = {key: [] for key in feats.keys()}
        for w in s:
            added = {key:False for key in feats.keys()}
            for f in w:
                if f == '<PAD>':
                    for key in feats.keys():
                        inputs[key].append(0)
                        added[key]=True
                    break
                try:
                    index = f.find('=')
                    key = f[:index]
                    value = f[index + 1:]
                    inputs[key].append(feats[key][value])
                    added[key]=True
                except KeyError:
                    inputs[key].append(1)
                    added[key]=True
                except ValueError:
                    pass
            for key in feats.keys():
                if added[key]==False:
                    inputs[key].append(1)

        for key in feats.keys():
            X[key].append(np.array(inputs[key]))

    return X


def encode_labels(y, idx):  # Kevin
    max_length = idx['max_length']
    labels = idx['labels']
    Y = []
    for s in y:
        outputs = []
        if len(s) <= max_length:
            s = s + ['<PAD>'] * (max_length - len(s))
        if len(s) > max_length:
            s = s[:max_length]
        for l in s:
            outputs.append([labels[l]])
        Y.append(np.array(outputs))

    return tf.keras.utils.to_categorical(Y)


def save_model_and_indexs(model, idx, filename):  # Kevin
    """
    Task: Save given model and indexs to disk
    Input:
        - model: Keras model created by build_network, and trained.
        - idx: A dictionary produced by create_indexs, containing word and label
               indexes, as well as the maximum sentence length.
        - filename: filename to be created.
    Output:
        Saves the model into filename.nn and the indexes into filename.idx
    """

    model.save(f'{filename}.nn.h5')
    file = open(f"{filename}.idx", 'wb')
    pickle.dump(idx, file)
    file.close()


def learn(traindir, validationdir, modelname):  # Ruizhe & Kevin
    """
    Learns a NN Model using traindir as training data, and validationdir
    as validation data. Saves learnt model in a file named modelname.
    """
    # load train and validation data in a suitable form.
    words_train, tags_train, sids_train, gt_train, offsets_train = load_data(traindir)
    words_val, tags_val, sids_val, gt_val, offsets_val = load_data(validationdir)

    # create indexes from training data
    max_len = 100
    idx = create_indexs(words_train, tags_train, max_length=max_len)

    # build network
    model = build_network(idx)

    # encode datasets
    feats_train=encode_words(words_train, idx)
    selected=feats_train.keys()
    Xtrain = np.stack([np.array(feats_train[key]) for key in selected],axis=1)
    Ytrain = encode_labels(tags_train, idx)
    feats_val=encode_words(words_val, idx)
    Xval = np.stack([np.array(feats_val[key]) for key in selected],axis=1)
    Yval = encode_labels(tags_val, idx)

    # hyperparameters
    batch_size = 32
    epochs = 20

    # train model
    model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xval, Yval))

    # save model and indexs, for later use in prediction
    save_model_and_indexs(model, idx, modelname)


if __name__ == '__main__':
    random.seed(0)
    tf.random.set_seed(0)

    # This code may be needed for some GPUs
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    traindir = sys.argv[1]
    validationdir = sys.argv[2]
    modelname = sys.argv[3]
    learn(traindir, validationdir, modelname)
#DEVEL
# ------------------------------------------------------------------------------
# brand             122     17     238     139     360    87.8%   33.9%   48.9%
# drug             1606    196     307    1802    1913    89.1%   84.0%   86.5%
# drug_n              5     11      40      16      45    31.2%   11.1%   16.4%
# group             538    127     143     665     681    80.9%   79.0%   79.9%
# ------------------------------------------------------------------------------
# M.avg            -      -       -       -       -       72.3%   52.0%   57.9%
# ------------------------------------------------------------------------------
# m.avg            2271    351     728    2622    2999    86.6%   75.7%   80.8%
# m.avg(no class)  2367    255     632    2622    2999    90.3%   78.9%   84.2%

#TEST
# ------------------------------------------------------------------------------
# brand             114     12     174     126     288    90.5%   39.6%   55.1%
# drug             1745    204     375    1949    2120    89.5%   82.3%   85.8%
# drug_n              0      9      72       9      72    0.0%    0.0%    0.0%
# group             560    161     139     721     699    77.7%   80.1%   78.9%
# ------------------------------------------------------------------------------
# M.avg            -      -       -       -       -       64.4%   50.5%   54.9%
# ------------------------------------------------------------------------------
# m.avg            2419    386     760    2805    3179    86.2%   76.1%   80.8%
# m.avg(no class)  2535    270     644    2805    3179    90.4%   79.7%   84.7%
