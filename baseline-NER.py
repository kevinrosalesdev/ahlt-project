import sys

from eval import evaluator
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize


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
    return word_tokenize(s)


def extract_entities(s):
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
    return [{"name": "Ascorbic acid", "offset": "0-12", "type": "drug"},
            {"name": "aspirin", "offset": "15-21", "type": "brand"}]


if __name__ == '__main__':
    datadir = sys.argv[1]
    outfile = sys.argv[2]
    outf = open(outfile, "w")
    # process each file in directory
    for f in listdir(datadir):
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
            entities = extract_entities(tokens)
            # print sentence entities in format requested for evaluation
            for e in entities:
                print(sid + "|" + e["offset"] + "|" + e["name"] + "|" + e["type"], file=outf)
    # print performance score
    evaluator.evaluate("NER", datadir, outfile)
