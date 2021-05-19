import sys
import pickle
import random
import tensorflow as tf
import numpy as np

from os import listdir
from xml.dom.minidom import parse
from nerc.feature_extractor import tokenize, get_tag
from nerc.learner import read_features
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras import Model


def load_data(path, features=False):  # Ruizhe
    if features:
        x, y, sids, infos = read_features(path)
    else:
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
                    words.append([f'form={tokens[i][0]}'])
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
    max_len = idx['max_length']
    embedding = 40

    # create network layers
    inp = Input(shape=(max_len,))
    # ... add missing layers here ...

    model = Embedding(input_dim=n_words, output_dim=embedding, input_length=max_len, mask_zero=True)(inp)
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation='relu'))(model)
    out = Dense(n_labels, activation='softmax')(model)  # final output layer

    # create and compile model
    model = Model(inp, out)

    # set appropriate parameters (optimizer, loss, etc.)
    model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

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
            for f in w:
                if f == '<PAD>':
                    for key in feats.keys():
                        inputs[key].append(0)
                    break
                try:
                    index = f.find('=')
                    key = f[:index]
                    value = f[index+1:]
                    inputs[key].append(feats[key][value])
                except KeyError:
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
    Xtrain = np.array(encode_words(words_train, idx)['form'])
    Ytrain = encode_labels(tags_train, idx)
    Xval = np.array(encode_words(words_val, idx)['form'])
    Yval = encode_labels(tags_val, idx)

    # hyperparameters
    batch_size = 64
    epochs = 10

    # train model
    model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xval, Yval))

    # save model and indexs, for later use in prediction
    save_model_and_indexs(model, idx, modelname)


if __name__ == '__main__':
    random.seed(0)

    # This code may be needed for some GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    traindir = sys.argv[1]
    validationdir = sys.argv[2]
    modelname = sys.argv[3]
    learn(traindir, validationdir, modelname)
