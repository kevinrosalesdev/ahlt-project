from nerc.learner import read_features
from nerc.feature_extractor import *
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from itertools import chain


def load_data(path, features=False):  # Ruizhe
    if features:
        x, y, sids, infos = read_features(path)
    else:
        x = []
        y = []
        sids = []
        infos = []
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
                for i in range(0, len(tokens)):
                    words.append([f'form={tokens[i][0]}'])
                    # see if the token is part of an entity, and which part (B/I)
                    tag = get_tag(tokens[i], gold)
                    tags.append(tag)
                y.append(tags)
                x.append(words)
    return x, y, sids, infos


def create_indexs(x, y, max_length):  # Ruizhe
    feats_aux = {}
    feats = {}

    for f in set(chain(*chain(*x))):
        try:
            name, value = f.split('=')
        except ValueError:
            pass #for now I ignore true or false features
        try:
            feats_aux[name].add(value)
        except KeyError:
            feats_aux[name] = set(value)
    for name in feats_aux.keys():
        feats[name] = {}
        for i, value in enumerate(feats_aux[name]):
            feats[name][value] = i+2
    lb = LabelBinarizer()
    lb.fit(list(chain(*y)))

    idx = {'feats': feats,
           'labels': {l: i + 2 for i, l in enumerate(lb.classes_)},
           'max_length': max_length
           }
    for key in idx['feats'].keys():
        idx['feats'][key]['<PAD>'] = 0
        idx['feats'][key]['<UNK>'] = 1
    idx['labels']['<PAD>'] = 0
    idx['labels']['<UNK>'] = 1
    return idx


def build_network():  # Kevin
    pass


def encode_words(x, idx):  # Ruizhe
    max_length = idx['max_length']
    feats = idx['feats']
    X = {key:[] for key in feats.keys()}
    for sentence in x:
        if len(sentence) <= max_length:
            s = sentence + [['<PAD>']] * (max_length - len(sentence))
        if len(sentence) > max_length:
            s = sentence[:max_length]
        inputs = {key:[] for key in feats.keys()}
        for w in s:
            added = {key:False for key in feats.keys()}

            for f in w:
                if f=='<PAD>':
                    for key in feats.keys():
                        inputs[key].append(0)
                        added[key]=True
                    break
                try:
                    key,value=f.split('=')
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
            X[key].append(inputs[key])
    return X


def encode_labels(y, idx):  # Kevin
    max_length = idx['max_length']
    labels = idx['labels']
    Y = []
    for s in y:
        outputs = []
        if len(s) <= max_length:
            s = s + ['<PAD>'] * (max_length - len(labels))
        if len(s) > max_length:
            s = s[:max_length]
        for l in s:
            outputs.append([labels[l]])
        Y.append(outputs)
    return Y


def save_model_and_indexs():  # Kevin
    pass


def learn():  # Ruizhe & Kevin
    pass


if __name__ == '__main__':
    x, y, sids, infos = load_data('nerc/train.feat',True)
    #x, y, sids, infos = load_data('data/train')
    idx = create_indexs(x, y, 100)
    x_train = encode_words(x, idx)
    y_train = encode_labels(y, idx)
