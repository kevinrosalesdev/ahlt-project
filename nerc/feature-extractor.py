import re
import sys
import numpy as np

from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize, TreebankWordTokenizer as twt


def load_drug_bank():
    f = open("../resources/DrugBank.txt", 'r', encoding="utf8")
    lines = f.readlines()  # array of file lines
    drug_bank_names = []
    drug_bank_types = []
    for line in lines:
        line = line.strip().split("|")
        drug_bank_names.append(line[0].lower())
        drug_bank_types.append(line[1])
    return drug_bank_names, drug_bank_types


def load_hsdb():
    f = open("../resources/HSDB.txt", 'r')
    lines = f.readlines()  # array of file lines
    hsdb_names = [line.strip().lower() for line in lines]
    hsdb_types = ['drug'] * len(lines)
    return hsdb_names, hsdb_types


def use_external_resources(token, drug_bank_names, drug_bank_types):
    possible_types = drug_bank_types[np.where(drug_bank_names == token)]
    if np.where(possible_types == 'drug')[0].size != 0:
        return 'drug'
    if np.where(possible_types == 'group')[0].size != 0:
        return 'group'
    if np.where(possible_types == 'brand')[0].size != 0:
        return 'brand'

    return None


def tokenize(s):
    """
    Task:
        Given a sentence, calls nltk.tokenize to split it in
        tokens, and adds to each token its start / end offset
        in the original sentence.
    Input:
        s: string containing the text for one sentence
    Output:
        Returns a list of tuples (word, offsetFrom, offsetTo)
    Example:
        >> tokenize ("Ascorbic acid , aspirin , and the common cold.")
        [("Ascorbic",0,7), ("acid",9,12), ("," ,13 ,13),
         ("aspirin",15,21), (",",22,22), ("and",24,26), ("the",28,30),
         ("common",32,37), ("cold",39,42), (".",43,43)]
    """
    raw_list = list(zip(word_tokenize(s), list(twt().span_tokenize(s))))
    return [(token, offset_1, offset_2 - 1) for token, (offset_1, offset_2) in raw_list]


def extract_features(s, look_er=True):
    """
    Task:
        Given a tokenized sentence, return a feature vector for each token
    Input :
        s: A tokenized sentence (list of triples (word, offsetFrom, offsetTo))
    Output :
        A list of feature vectors, one per token.
        Features are binary and vectors are in sparse representation (i.e. only
        active features are listed)
    Example :
        >> extract_features ([("Ascorbic",0,7), ("acid",9,12), (",",13,13),
                               ("aspirin",15,21), (",",22,22), ("and",24,26), ("the",28,30),
                               ("common",32,37), ("cold",39,42), (".",43,43)])
            [["form=Ascorbic","suf4=rbic","next=acid","prev=_BoS_","capitalized"],
            ["form=acid","suf4=acid","next=,","prev=Ascorbic"],
            ["form=,","suf4=,","next=aspirin","prev=acid","punct"],
            ["form=aspirin","suf4=irin","next=,","prev=,"],
            ...]
    """
    output = []
    # pos_tags = [pos_tag[1] for pos_tag in pos_tag(word_tokenize(stext))]

    drug_bank_names, drug_bank_types = load_drug_bank()
    hsdb_names, hsdb_types = load_hsdb()
    drug_bank_names.extend(hsdb_names)
    drug_bank_types.extend(hsdb_types)
    drug_bank_names = np.array(drug_bank_names)
    drug_bank_types = np.array(drug_bank_types)


    for i, token in enumerate(s):
        features = []

        # Form
        features.append(f'form={token[0]}')

        # Suffix 5, 4, 3. The word length without suffix must be at least 2+len(suffix)
        if len(token[0]) >= 7:
            features.append(f'suf5={token[0][-5:]}')
        if len(token[0]) >= 6:
            features.append(f'suf4={token[0][-4:]}')
        if len(token[0]) >= 5:
            features.append(f'suf3={token[0][-3:]}')

        # Suffix 2 and 1 (last characters of word)
        if len(token[0]) >= 2:
            features.append(f'suf2={token[0][-2:]}')
        features.append(f'suf1={token[0][-1:]}')

        # Next token (No suffix information as it decreases the M.F1)
        if i != len(s) - 1:
            features.append(f'next={s[i + 1][0]}')
        else:
            features.append('next=_EoS_')

        # Previous token (No suffix information as it decreases the M.F1)
        if i != 0:
            features.append(f'prev={s[i - 1][0]}')
        else:
            features.append('prev=_BoS_')

        # Capitalized
        if token[0][0].isupper():
            features.append('capitalized')

        # With numbers
        if re.findall(r'[0-9]+', token[0]):
            features.append("number")

        # With dashes
        # if re.findall(r'-+', token[0]):
        #     features.append("dash")

        # Length
        features.append(f'length={len(token[0])}')

        # Pos Tag
        # features.append(f'postag={pos_tags[i]}')

        if look_er:
            type = use_external_resources(token[0].lower(), drug_bank_names, drug_bank_types)
            if type is not None:
                features.append(f'ertype={type}')

        output.append(features)

    return output


def get_tag(token, gold):
    """
    Task:
        Given a token and a list of ground truth entities in a sentence, decide
        which is the B-I-O tag for the token
    Input:
        token: A token, i.e. one triple (word, offsetFrom, offsetTo)
        gold: A list of ground truth entities, i.e. a list of triples (offsetFrom, offsetTo, type)
    Output:
        The B-I-O ground truth tag for the given token ("B-drug","I-drug","B-group","I-group","O", ...)
    Example:
        >> get_tag (("Ascorbic",0,7), [(0, 12,"drug"), (15, 21,"brand") ])
        B-drug
        >> get_tag (("acid",9,12), [(0, 12,"drug"), (15, 21,"brand") ])
        I-drug
        >> get_tag (("common",32,37), [(0, 12,"drug"), (15, 21,"brand") ])
        O
        >> get_tag (("aspirin",15,21), [(0, 12,"drug"), (15, 21,"brand") ])
        B-brand
    """
    for gte in gold:
        if token[1] >= gte[0] and token[2] <= gte[1]:
            if token[1] == gte[0]:
                return "B-" + gte[2]
            return "I-" + gte[2]

    return "O"


if __name__ == '__main__':
    datadir = sys.argv[1]

    # process each file in directory
    for idx, f in enumerate(listdir(datadir), 1):

        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # load ground truth entities
            gold = []
            entities = s.getElementsByTagName("entity")
            for e in entities:
                # for discontinuous entities, we only get the first pan
                offset = e.attributes["charOffset"].value
                (start, end) = offset.split(";")[0].split("-")
                gold.append((int(start), int(end), e.attributes["type"].value))

            # tokenize text
            tokens = tokenize(stext)
            # extract features for each word in the sentence
            features = extract_features(tokens)
            # print features in format suitable for the learner/classifier
            for i in range(0, len(tokens)):
                # see if the token is part of an entity, and which part (B/I)
                tag = get_tag(tokens[i], gold)
                print(sid, tokens[i][0], tokens[i][1], tokens[i][2],
                      tag, "\t".join(features[i]), sep='\t')
            print()
