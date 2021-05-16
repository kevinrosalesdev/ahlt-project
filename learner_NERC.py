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
    mlb = MultiLabelBinarizer()
    mlb.fit([['<PAD>']]+list(chain(*x)))
    lb = LabelBinarizer()
    lb.fit(['<PAD>']+list(chain(*y)))
    return {'feats': mlb,
            'labels': lb,
            'max_length': max_length
            }


def build_network():  # Kevin
    pass


def encode_words(x, idx):  # Ruizhe
    max_length=idx['max_length']
    feats_mlb=idx['feats']
    X=[]
    for sentence in x:
        if len(sentence) <= max_length:
            s = sentence + [['<PAD>']] * (max_length - len(sentence))
        if len(sentence) > max_length:
            s = sentence[:max_length]
        s=feats_mlb.transform(s)
        X.append(s)
    return X


def encode_labels(y):  # Kevin
    max_length=idx['max_length']
    labels_lb=idx['labels']
    Y=[]
    for labels in y:
        if len(labels) <= max_length:
            l = labels + [['<PAD>']] * (max_length - len(labels))
        if len(labels) > max_length:
            l = labels[:max_length]
        l=labels_lb.transform(l)
        Y.append(l)
    return Y


def save_model_and_indexs():  # Kevin
    pass


def learn():  # Ruizhe & Kevin
    pass
