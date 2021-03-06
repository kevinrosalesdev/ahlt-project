import sys
import pycrfsuite
from learner import read_features

def output_entities(sid, tokens, tags):
    """
    Task:
        Given a list of tokens and the B-I-O tag for each token, produce a list
        of drugs in the format expected by the evaluator.
    Input:
        sid: sentence identifier (required by the evaluator output format)
        tokens: List of tokens in the sentence, i.e. list of tuples (word, offsetFrom, offsetTo)
        tags: List of B-I-O tags for each token
    Output:
        Prints to stdout the entities in the right format: one line per entity,
        fields separated by ’| ’, field order: id, offset, name, type.
    Example:
        >> output_entities ("DDI-DrugBank.d553.s0",
                            [("Ascorbic",0,7), ("acid",9,12), (",",13,13),
                            ("aspirin",15,21), (",",22,22), ("and",24,26),
                            ("the",28,30),("common",32,37), ("cold",39,42)],
                            ["B-drug","I-drug","O","B-brand","O","O","O",
                             "O","O"])
        DDI-DrugBank.d553.s0|0-12|Ascorbic acid|drug
        DDI-DrugBank.d553.s0|15-21|aspirin|brand
    """
    last_found = 'O'
    for index in range(len(tokens)):
        if tags[index][0] == 'B':
            # B-tag & the last tag was different to O-tag -> New entity must be printed.
            if last_found != 'O':
                print(f"{sid}|{offset[0]}-{offset[1]}|{name}|{type}")

            name = tokens[index][0]
            offset = [tokens[index][1], tokens[index][2]]
            type = tags[index][2:]
            last_found = 'B'

        elif tags[index][0] == 'I':
            name += " " + tokens[index][0]
            offset[1] = tokens[index][2]
            last_found = 'I'

        else:
            # O-tag & the last tag was different to O-tag -> New entity must be printed.
            if last_found != 'O':
                print(f"{sid}|{offset[0]}-{offset[1]}|{name}|{type}")
            last_found = 'O'

    # If the last tag of the array was different from O, it must be printed too.
    if last_found != 'O':
        print(f"{sid}|{offset[0]}-{offset[1]}|{name}|{type}")


if __name__ == '__main__':
    # output_entities("DDI-DrugBank.d553.s0",
    #                 [("Ascorbic", 0, 7), ("acid", 9, 12),
    #                  ("aspirin", 15, 21), (",", 22, 22), ("and", 24, 26),
    #                  ("the", 28, 30), ("common", 32, 37), ("cold", 39, 42)],
    #                 ["B-drug", "I-drug", "B-brand", "O", "O", "O", "O", "O"])

    model_name = sys.argv[1]
    features = sys.argv[2]

    tagger = pycrfsuite.Tagger()
    tagger.open(f'{model_name}.crfsuite')
    x,y,sids,infos=read_features(features)
    for feats,labels,sid,info in zip(x,y,sids,infos):
        output_entities(sid,info,tagger.tag(feats))

