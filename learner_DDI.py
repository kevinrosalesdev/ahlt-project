import random, sys, pickle, os

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from itertools import chain
from xml.dom.minidom import parse
from os import listdir
from nltk import CoreNLPDependencyParser
from sklearn.preprocessing import LabelEncoder


def get_entity_idx(analysis: dict, offset):
    # Some GT offsets are ""215-226;246-276" because the drug is not consecutive
    if len(offset) > 2:
        offset = [offset[0], offset[-1]]

    idx_e = [0, 10]

    for idx in range(1, len(analysis.keys())+1):
        for shift in range(0, 10):
            if idx_e[1] > shift and analysis[idx]['offset'][0] >= int(offset[0]) - shift \
                    and analysis[idx]['offset'][1] <= int(offset[1]) + shift:
                idx_e = [idx, shift]

    return idx_e[0]


def analyze(s, parser, entities, id_e1, id_e2):
    s = s.replace("%", "%25").replace("+", "%2B").replace("\r\n", ", ").replace(".", ",")
    if len(s) > 0:
        props = {'annotators': 'pos, lemma',
                 'pipelineLanguage': 'en',
                 'outputFormat': 'json'}

        parsed_str = parser.api_call(s, properties=props)['sentences'][0]['tokens']
        res = {}
        for token in parsed_str:
            res[token['index']] = {'word': token['word'],
                                   'lemma': token['lemma'],
                                   'offset': [token['characterOffsetBegin'], token['characterOffsetEnd']-1],
                                   'pos': token['pos'],
                                   'mask': None}
        for e in entities.keys():
            offset = entities[e]
            if e == id_e1:
                res[get_entity_idx(res, offset)]['mask'] = '<DRUG1>'
            elif e == id_e2:
                res[get_entity_idx(res, offset)]['mask'] = '<DRUG2>'
            else:
                res[get_entity_idx(res, offset)]['mask'] = '<DRUG_OTHER>'

        return res
    else:
        print("Empty sentence!")
        return {}


def load_data(dir, load_pickle=None):
    """
    Task:
        Load XML Files in given directory, tokenize each sentence, and extract
        learning examples (tokenized sentence + entity pair)

    Input:
        datadir: A directory containing XML Files

    Output:
        A list of classification cases. Each case is a list containing sentence id,
        entity1 id, entity2 id, ground truth relation label, and a list
        of sentence tokens (each token containing any needed information: word,
        lemma, PoS, offsets, etc.)
    """
    if load_pickle and os.path.exists(f'tmp_{load_pickle}_data.pickle'):
        f = open(f'tmp_{load_pickle}_data.pickle', 'rb')
        data = pickle.load(f)
        f.close()
        return data


    classification_cases = []
    my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

    # process each file in directory
    for idx, f in enumerate(listdir(dir), 1):
        print(f"Processing file nº {idx}/{len(listdir(dir))}")

        # parse XML file, obtaining a DOM tree
        tree = parse(dir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text

            # load sentence entities into dictionary
            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                eid = e.attributes["id"].value
                entities[eid] = e.attributes["charOffset"].value.split("-")

            # for each pair in the sentence, decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                ddi = p.attributes["ddi"].value
                dditype = p.attributes["type"].value if ddi == 'true' else 'null'

                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value

                analysis = analyze(stext, my_parser, entities, id_e1, id_e2)

                classification_cases.append((sid, id_e1, id_e2, dditype, analysis))
                # print((sid, id_e1, id_e2, dditype, analysis), "\n")

    if not load_pickle:
        f = open(f'tmp_{load_pickle}_data.pickle', 'wb')
        pickle.dump(classification_cases, f)
        f.close()

    return classification_cases


def create_indexs(dataset, max_length):
    """
    Task:
        Create index dictionaries both for input (words) and outputs (labels)
        from given dataset.

    Input:
        dataset: dataset produced by load_data.
        max_length: maximum length of a sentence (longer sentences will be cut,
                    shorter ones will be padded).

    Output:
        A dictionary where each key is an index name (e.g. 'words', 'labels'),
        and the value is a dictionary mapping each word/label to a number.
        An entry with the value for maxlen is also stored.

    Example:
        >> create_indexs(traindata)
        {'word': {'<PAD>': 0, '<UNK>': 1, '11-day': 2, 'murine': 3, 'criteria': 4
                    ... 'terfenadine': 8512}
        'labels': {'null': 0, 'mechanism': 1, 'advise': 2, 'effect': 3, 'int': 4}
        'maxlen': 100}
    """
    idxs = {}

    x = [[case[4][i]['word'] if not case[4][i]['mask'] else case[4][i]['mask'] for i in range(1, len(case[4].keys())+1)]
         for case in dataset]
    x = list(chain(*x))
    le = LabelEncoder()
    le.fit(x)
    idxs['word'] = {i: le.transform([i])[0]+2 for i in le.classes_}
    idxs['word']['<PAD>'] = 0
    idxs['word']['<UNK>'] = 1

    x = [[case[4][i]['lemma'] if not case[4][i]['mask'] else case[4][i]['mask'] for i in range(1, len(case[4].keys())+1)]
         for case in dataset]
    x = list(chain(*x))
    le = LabelEncoder()
    le.fit(x)
    idxs['lemma'] = {i: le.transform([i])[0]+2 for i in le.classes_}
    idxs['lemma']['<PAD>'] = 0
    idxs['lemma']['<UNK>'] = 1

    x = [[case[4][i]['pos'] if not case[4][i]['mask'] else case[4][i]['mask'] for i in range(1, len(case[4].keys())+1)]
         for case in dataset]
    x = list(chain(*x))
    le = LabelEncoder()
    le.fit(x)
    idxs['pos'] = {i: le.transform([i])[0]+2 for i in le.classes_}
    idxs['pos']['<PAD>'] = 0
    idxs['pos']['<UNK>'] = 1

    y = [case[3] for case in dataset]
    le = LabelEncoder()
    le.fit(y)
    idxs['labels'] = {i: le.transform([i])[0] for i in le.classes_}

    idxs['maxlen'] = max_length

    return idxs


def build_network(idx):
    """
    Task: Create network for the learner
    Input:
        idx: index dictionary with words/labels codes, plus maximum sentence length
    Output:
        Returns a compiled Keras neural network with the specified layers.
    """

    # sizes
    n_words = len(idx['word'])
    n_labels = len(idx['labels'])
    max_len = idx['maxlen']

    embedding = 64

    # create network layers
    inp = Input(shape=(3, max_len, ))
    # ... add missing layers here ...

    model = Embedding(input_dim=n_words, output_dim=embedding, mask_zero=True)(inp)
    model = Conv1D(128, 5, activation='relu')(model)
    model = GlobalMaxPooling2D()(model)
    model = Dense(10, activation='relu')(model)
    out = Dense(n_labels, activation='softmax')(model)  # final output layer

    # create and compile model
    model = Model(inp, out)

    # set appropriate parameters (optimizer, loss, etc.)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def encode_words(dataset, idx):
    """
    Task:
        Encode the words in a sentence dataset formed by lists of tokens into lists of indexes
        suitable for NN input.

    Input:
        dataset: A dataset produced by load_data.
        idx: A dictionary produced by create_indexs, containing word and label indexes, as well
             as the maximum sentence length.

    Output:
        The dataset encoded as a list of sentence, each of them is a list of words indices. If the
        word is not in the index, <UNK> code is used. If the sentence is shorter than max_len it is
        padded with <PAD> code.

    Example:
        >> encode_words(traindata, idx)
        [[6882 1049 4911 ... 0 0 0]
         [........................]
         [2002 6582 7518 ... 0 0 0]]
    """
    nn_input = []
    for case in dataset:
        sentence = []
        for info in ['word', 'lemma', 'pos']:
            sentence_info = []
            for token in case[4].keys():
                if case[4][token]['mask']:
                    sentence_info.append(idx[info][case[4][token]['mask']])
                else:
                    try:
                        sentence_info.append(idx[info][case[4][token][info]])
                    except:
                        sentence_info.append(idx[info]['<UNK>'])

            if len(sentence_info) <= idx['maxlen']:
                sentence_info.extend([idx[info]['<PAD>']] * (idx['maxlen'] - len(sentence_info)))
            if len(sentence_info) > idx['maxlen']:
                sentence_info = sentence_info[:idx['maxlen']]

            sentence.append(sentence_info)

        nn_input.append(sentence)

    return np.array(nn_input)


def encode_labels(dataset, idx):
    """
    Task:
        Encode the ground truth labels in a dataset of classification examples
        (sentence + entity pair).

    Input:
        dataset: A dataset produced by load_data.
        idx: A dictionary produced by create_indexs, containing word and label indexes,
             as well as the maximum sentence length.

    Output:
        The dataset encoded as a list DDI labels, one per classification example.

    Example:
        >> encode_labels(traindata, idx)
        [ [0] [0] [2] ... [0] [1] [0] ]
    """
    Y = [[idx['labels'][case[3]]] for case in dataset]
    return tf.keras.utils.to_categorical(Y)


def save_model_and_indexs(model, idx, filename):
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


def learn(traindir, validationdir, modelname):
    """
    Learns a NN Model using traindir as training data, and validationdir
    as validation data. Saves learnt model in a file named modelname.
    """
    # load train and validation data in a suitable form.
    train_data = load_data(traindir, load_pickle='train')
    val_data = load_data(validationdir, load_pickle='val')

    # create indexes from training data
    max_len = 100
    idx = create_indexs(train_data, max_length=max_len)

    # build network
    model = build_network(idx)

    # encode datasets
    Xtrain = encode_words(train_data, idx)
    Ytrain = encode_labels(train_data, idx)
    Xval = encode_words(val_data, idx)
    Yval = encode_labels(val_data, idx)

    # hyperparameters
    batch_size = 64
    epochs = 15

    # train model
    model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xval, Yval))

    # save model and indexs, for later use in prediction
    save_model_and_indexs(model, idx, modelname)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    # This code may be needed for some GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    traindir = sys.argv[1]
    validationdir = sys.argv[2]
    modelname = sys.argv[3]
    learn(traindir, validationdir, modelname)
