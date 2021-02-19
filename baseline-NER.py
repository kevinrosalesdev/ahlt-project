import sys

from eval import evaluator
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize, TreebankWordTokenizer as twt


def load_drug_bank():
    print("Loading Drug Bank...")
    f = open("resources/DrugBank.txt", 'r')
    lines = f.readlines()  # array of file lines
    drug_bank_names = []
    drug_bank_types = []
    for line in lines:
        line = line.strip().split("|")
        drug_bank_names.append(line[0])
        drug_bank_types.append(line[1])
    return drug_bank_names, drug_bank_types


def load_hsdb():
    print("Loading HSDB...")
    f = open("resources/HSDB.txt", 'r')
    lines = f.readlines()  # array of file lines
    hsdb_names = [line.strip() for line in lines]
    hsdb_types = ['drug']*len(lines)
    return hsdb_names, hsdb_types


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


def use_external_resources(token, drug_bank_names, drug_bank_types):
    try:
        return drug_bank_types[drug_bank_names.index(token)]
    except:
        return 'drug'


def extract_entities(s, drug_bank_names, drug_bank_types):
    """
    Task:
        Given a tokenized sentence, identify which tokens (or groups of
        consecutive tokens) are drugs
    Input :
        s: A tokenized sentence ( list of triples (word , offsetFrom , offsetTo ) )
    Output :
        A list of entities. Each entity is a dictionary with the keys ’name’, ’offset', and ’type’.
    Example :
        >> extract_entities([("Ascorbic",0,7), ("acid",9,12), ("," ,13 ,13),
         ("aspirin",15,21), (",",22,22), ("and",24,26), ("the",28,30),
         ("common",32,37), ("cold",39,42), (".",43,43)])
        [{"name":"Ascorbic acid", "offset":"0-12", "type":"drug"},
        {"name":"aspirin", "offset":"15-21", "type":"brand"}]
    """
    entities = []
    for token in s:
        # TODO: Chain of Rules filtering & classifying tokens should be here!
        type = use_external_resources(token[0], drug_bank_names, drug_bank_types)
        entities.append({"name": token[0],
                         "offset": f"{token[1]}-{token[2]}",
                         "type": type})

    return entities


if __name__ == '__main__':
    datadir = sys.argv[1]
    outfile = sys.argv[2]
    outf = open(outfile, "w")
    drug_bank_names, drug_bank_types = load_drug_bank()
    hsdb_names, hsdb_types = load_hsdb()
    drug_bank_names.extend(hsdb_names)
    drug_bank_types.extend(hsdb_types)

    # process each file in directory
    for idx, f in enumerate(listdir(datadir), 1):
        print(f"Processing file nº {idx}/{len(listdir(datadir))}")
        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # tokenize text
            tokens = tokenize(stext)
            # extract entities from tokenized sentence text
            entities = extract_entities(tokens, drug_bank_names, drug_bank_types)
            # print sentence entities in format requested for evaluation
            for e in entities:
                print(sid + "|" + e["offset"] + "|" + e["name"] + "|" + e["type"], file=outf)
    # print performance score
    evaluator.evaluate("NER", datadir, outfile)
