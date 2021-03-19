import sys
import re
import numpy as np

from nltk.corpus import stopwords

import evaluator
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize, TreebankWordTokenizer as twt


def load_drug_bank():
    print("Loading Drug Bank...")
    f = open("resources/DrugBank.txt", 'r', encoding="utf8")
    lines = f.readlines()  # array of file lines
    drug_bank_names = []
    drug_bank_types = []
    for line in lines:
        line = line.strip().split("|")
        drug_bank_names.append(line[0].lower())
        drug_bank_types.append(line[1])
    return drug_bank_names, drug_bank_types


def load_hsdb():
    print("Loading HSDB...")
    f = open("resources/HSDB.txt", 'r')
    lines = f.readlines()  # array of file lines
    hsdb_names = [line.strip().lower() for line in lines]
    hsdb_types = ['drug'] * len(lines)
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
    possible_types = drug_bank_types[np.where(drug_bank_names == token)]
    if np.where(possible_types == 'drug')[0].size != 0:
        return 'drug'
    if np.where(possible_types == 'group')[0].size != 0:
        return 'group'
    if np.where(possible_types == 'brand')[0].size != 0:
        return 'brand'

    return None


def extract_entities(s, drug_bank_names, drug_bank_types, stopwords, look_er=True):
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
    for i, token in enumerate(s):
        # Use external resources
        # print(f"Index:{i+1}/{len(s)} -> {token}")
        if look_er:
            type = use_external_resources(token[0].lower(), drug_bank_names, drug_bank_types)
            if type is not None:
                print(f"\tReason: ER ({type})", token[0])
                entities.append({"name": token[0],
                                 "offset": f"{token[1]}-{token[2]}",
                                 "type": type})
                continue

            if i + 1 < len(s):
                glued_token = (tokens[i][0] + " " + tokens[i + 1][0], tokens[i][1], tokens[i + 1][2])
                type = use_external_resources(glued_token[0].lower(), drug_bank_names, drug_bank_types)
                if type is not None:
                    print(f"\tReason: ER (GLUED) ({type})", glued_token[0])
                    entities.append({"name": glued_token[0],
                                     "offset": f"{glued_token[1]}-{glued_token[2]}",
                                     "type": type})
                    continue

            if i + 2 < len(s):
                glued_token = (tokens[i][0] + " " + tokens[i + 1][0] + " " + tokens[i + 2][0],
                               tokens[i][1], tokens[i + 2][2])
                type = use_external_resources(glued_token[0].lower(), drug_bank_names, drug_bank_types)
                if type is not None:
                    print(f"\tReason: ER (GLUED) ({type})", glued_token[0])
                    entities.append({"name": glued_token[0],
                                     "offset": f"{glued_token[1]}-{glued_token[2]}",
                                     "type": type})
                    continue

        if len(token[0]) <= 5 or len(re.findall(r'[A-Z|a-z]', token[0])) <= 3 or token[0].lower() in stopwords:
            continue

        # If suffix found in list assign group
        group_suffixes = ['oid', 'oids', 'osides$']
        if re.findall("$|".join(group_suffixes), token[0]):
            print("\tReason: Suffix (Group)", token[0])
            entities.append({"name": token[0],
                             "offset": f"{token[1]}-{token[2]}",
                             "type": 'group'})
            continue

        # If suffix found in list, assign drug
        drug_suffixes = ['acin', 'acin', 'adrine', 'afil', 'afine', 'aine', 'aline', 'alone', 'amide', 'amil', 'amine',
                         'ampin', 'apine', 'apine', 'apride', 'arin', 'avir', 'azine', 'azole', 'azone', 'bital',
                         'cycline', 'drone', 'eine', 'emide', 'epine', 'esin', 'etate', 'etin', 'etine', 'etine', 'fen',
                         'icin', 'idine', 'ipin', 'ipine', 'isone', 'itone', 'lide', 'llin', 'lline', 'micin',
                         'mycin', 'ocine', 'odone', 'nol', 'hol', 'olimus', 'olin', 'oline', 'olone', 'omide', 'onide',
                         'orin',
                         'osine', 'otine', 'oxacin', 'oxib', 'oxide', 'oxin', 'oxone', 'phine', 'phrine',
                         'rtan', 'strel', 'tatin', 'taxel', 'thane', 'udine', 'ulin', 'utin', 'ycine', 'zepam', 'zine',
                         'atrin',
                         'afil', 'toin', 'oprim', 'axel$']
        if re.findall("$|".join(drug_suffixes), token[0]):
            print("\tReason: Suffix (Drug)", token[0])
            entities.append({"name": token[0],
                             "offset": f"{token[1]}-{token[2]}",
                             "type": 'drug'})
            continue


        # If word is fully capitalized, assign brand
        if token[0].isupper():
            print("\tReason: Supper (Brand)", token[0])
            entities.append({"name": token[0],
                             "offset": f"{token[1]}-{token[2]}",
                             "type": 'brand'})
            continue

    return entities


if __name__ == '__main__':
    datadir = sys.argv[1]
    outfile = sys.argv[2]
    outf = open(outfile, "w")
    sw = set(stopwords.words('english'))
    drug_bank_names, drug_bank_types = load_drug_bank()
    hsdb_names, hsdb_types = load_hsdb()
    drug_bank_names.extend(hsdb_names)
    drug_bank_types.extend(hsdb_types)
    drug_bank_names = np.array(drug_bank_names)
    drug_bank_types = np.array(drug_bank_types)

    # process each file in directory
    for idx, f in enumerate(listdir(datadir), 1):
        print(f"Processing file nº {idx}/{len(listdir(datadir))}")

        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        # print("File:", f)
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # print("Sentence:", stext)
            # tokenize text
            tokens = tokenize(stext)
            # extract entities from tokenized sentence text
            entities = extract_entities(tokens, drug_bank_names, drug_bank_types, sw)
            # print sentence entities in format requested for evaluation
            for e in entities:
                print(sid + "|" + e["offset"] + "|" + e["name"] + "|" + e["type"], file=outf)
    outf.close()
    # print performance score
    evaluator.evaluate("NER", datadir, outfile)
