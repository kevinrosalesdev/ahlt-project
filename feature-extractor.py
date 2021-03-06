import sys

from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize, TreebankWordTokenizer as twt


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


# TODO
def extract_features(s):
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
    output=[]
    slen=len(s)
    for i,token in enumerate(s):
        features=[]

        #Form
        features.append(f'form={token[0]}')

        #Suffix 4
        features.append(f'suf4={token[0][-4:]}')

        #Next token
        if i!=slen-1:
            features.append(f'next={s[i+1][0]}')
        else:
            features.append('next=_EoS_')

        #Previous token
        if i!=0:
            features.append(f'prev={s[i-1][0]}')
        else:
            features.append('prev=_BoS_')

        #Capitalized

        if token[0][0].isupper():
            features.append('capitalized')

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
