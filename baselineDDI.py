import sys
import evaluator
import networkx as nx

from os import listdir
from xml.dom.minidom import parse
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import DependencyGraph


def analyze(s, parser):
    """
    Task:
        Given one sentence, sends it to CoreNLP to obtain the tokens, tags, and dependency tree.
        It also adds the start/end offsets to each token.
    Input:
        s: string containing the text for one sentence
    Output:
        Returns the nltk DependencyGraph (https://www.nltk.org/_modules/nltk/parse/dependencygraph.html)
        object produced by CoreNLP, enriched with token offsets.
    Example:
    >> analyze ("Caution should be exercised when combining resorcinol or salicylic acid with DIFFERIN Gel")
        {0:{’head’:None,’lemma’:None,’rel’:None,’tag’:’TOP’,’word’: None},
        1:{’word’:’Caution’,’head’:4,’lemma’:’caution’,’rel’:’nsubjpass’,’tag’:’NN’,’start’:0,’end’:6},
        2:{’word’:’should’,’head’:4,’lemma’:’should’,’rel’:’aux’,’tag’:’MD’,’start’:8,’end’:13},
        3:{’word’:’be’,’head’:4,’lemma’:’be’,’rel’:’auxpass’,’tag’:’VB’,’start’:15,’end’:16},
        4:{’word’:’exercised’,’head’:0,’lemma’:’exercise’,’rel’:’ROOT’,’tag’:’VBN’,’start’:18,’end’:26},
        5:{’word’:’when’,’head’:6,’lemma’:’when’,’rel’:’advmod’,’tag’:’WRB’,’start’:28,’end’:31},
        6:{’word’:’combining’,’head’:4,’lemma’:’combine’,’rel’:’advcl’,’tag’:’VBG’,’start’:33,’end’:41},
        7:{’word’:’resorcinol’,’head’:6,’lemma’:’resorcinol’,’rel’:’dobj’,’tag’:’NN’,’start’:43,’end’:52},
    """

    # '%' is a reserved token for CoreNLP
    s = s.replace("%", "")
    if len(s) > 0:
        # parse text (as many times as needed).
        mytree, = parser.raw_parse(s)  # type: DependencyGraph
        offsets = parser.api_call(s, properties={'annotators': 'tokenize'})['tokens']
        for idx in range(len(offsets)):
            mytree.nodes[idx + 1]['start'] = offsets[idx]['characterOffsetBegin']
            mytree.nodes[idx + 1]['end'] = offsets[idx]['characterOffsetEnd'] - 1
        return mytree
    else:
        print("Empty sentence!")
        return {}


def write_path(analysis, idx_1, idx_2, idx):
    path1 = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_1, idx)
    path2 = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_2, idx)
    path = analysis.nodes[idx]['word']
    for i in path1[:-1][::-1]:
        path = f"{analysis.nodes[i]['word']} [{analysis.nodes[i]['rel']}] > {path}"
    for i in path2[:-1][::-1]:
        path = f"{path} < [{analysis.nodes[i]['rel']}] {analysis.nodes[i]['word']}"

    path1, path2 = path.split(f"> {analysis.nodes[idx]['word']} <")
    # print(path)
    return path, path1, path2


def get_list(lemma, lists):
    if lemma in lists[0]:
        return 'effect'
    elif lemma in lists[1]:
        return 'mechanism'
    elif lemma in lists[2]:
        return 'int'
    elif lemma in lists[3]:
        return 'advise'
    return False


def check_interaction(analysis: DependencyGraph, entities, e1, e2):
    """
    Task:
        Decide whether a sentence is expressing a DDI between two drugs.
    Input:
        analysis: a DependencyGraph object with all sentence information
        entities: A list of all entities in the sentence (id and offsets)
        e1, e2: ids of the two entities to be checked.
    Output:
        Returns the type of interaction (’effect’, ’mechanism’, ’advice’, ’int’)
        between e1 and e2 expressed by the sentence, or ’None’ if no interaction is described.
    """
    offset_1 = entities[e1]
    offset_2 = entities[e2]
    # Some GT offsets are ""215-226;246-276" because the drug is not consecutive
    if len(offset_1) > 2:
        offset_1 = [offset_1[0], offset_1[-1]]
    if len(offset_2) > 2:
        offset_2 = [offset_2[0], offset_2[-1]]

    idx_1 = [0, 10]
    idx_2 = [0, 10]

    for idx in range(1, len(analysis.nodes)):
        for shift in range(0, 10):
            if idx_1[1] > shift and analysis.nodes[idx]['start'] >= int(offset_1[0]) - shift \
                    and analysis.nodes[idx]['end'] <= int(offset_1[1]) + shift:
                idx_1 = [idx, shift]
            if idx_2[1] > shift and analysis.nodes[idx]['start'] >= int(offset_2[0]) - shift \
                    and analysis.nodes[idx]['end'] <= int(offset_2[1]) + shift:
                idx_2 = [idx, shift]

    # Rules
    idx_1 = idx_1[0]
    idx_2 = idx_2[0]

    effects_list = ['administer', 'potentiate', 'prevent', 'block', 'cause', 'enhance']
    mech_list = ['reduce', 'increase', 'decrease']
    int_list = ['interact', 'interaction']
    advise_list = ['advise', 'recommend', 'caution', 'consider']

    total_list = [effects_list, mech_list, int_list, advise_list]

    # effect
    for idx in range(1, len(analysis.nodes)):
        if analysis.nodes[idx]['lemma'] == 'should':
            try:
                head = analysis.nodes[idx]['head']
                write_path(analysis, idx_1, idx_2, head)
                return 'advise'
            except nx.exception.NetworkXNoPath:
                continue

    for idx in range(1, len(analysis.nodes)):
        lemma_list = get_list(analysis.nodes[idx]['lemma'], total_list)
        try:
            if lemma_list:
                path, path1, path2 = write_path(analysis, idx_1, idx_2, idx)
                if ('[nsubj]' in path1) & ('[obj]' in path2):
                    return lemma_list
        except nx.exception.NetworkXNoPath:
            continue

    for idx in range(1, len(analysis.nodes)):
        lemma_list = get_list(analysis.nodes[idx]['lemma'], total_list)
        try:
            if lemma_list:
                write_path(analysis, idx_1, idx_2, idx)
                return lemma_list
        except nx.exception.NetworkXNoPath:
            continue

    return None


if __name__ == '__main__':
    inputdir = sys.argv[1]
    outputfile = sys.argv[2]
    outf = open(outputfile, "w")

    # connect to your CoreNLP server (just once)
    my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

    # process each file in directory
    for idx, f in enumerate(listdir(inputdir), 1):
        print(f"Processing file nº {idx}/{len(listdir(inputdir))}")

        # parse XML file, obtaining a DOM tree
        tree = parse(inputdir + "/" + f)
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

            # Tokenize, tag, and parse sentence
            analysis = analyze(stext, my_parser)

            # for each pair in the sentence, decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                ddi_type = check_interaction(analysis, entities, id_e1, id_e2)
                if ddi_type is not None:
                    print(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi_type, file=outf)

    outf.close()
    # get performance score
    evaluator.evaluate("DDI", inputdir, outputfile)

# if __name__ == '__main__':
#     inputdir = 'data/train'
#     label = 'mechanism'
#
#     # connect to your CoreNLP server (just once)
#     my_parser = CoreNLPDependencyParser(url="http://localhost:9000")
#
#     # process each file in directory
#     for idx, f in enumerate(listdir(inputdir), 1):
#         print(f"Processing file nº {idx}/{len(listdir(inputdir))}")
#
#         # parse XML file, obtaining a DOM tree
#         tree = parse(inputdir + "/" + f)
#         print(f)
#         # process each sentence in the file
#         sentences = tree.getElementsByTagName("sentence")
#         for s in sentences:
#             sid = s.attributes["id"].value  # get sentence id
#             stext = s.attributes["text"].value  # get sentence text
#
#             # load sentence entities into dictionary
#             entities = {}
#             ents = s.getElementsByTagName("entity")
#             for e in ents:
#                 eid = e.attributes["id"].value
#                 entities[eid] = e.attributes["charOffset"].value.split("-")
#
#             # Tokenize, tag, and parse sentence
#             analysis = analyze(stext, my_parser)
#
#             # for each pair in the sentence, decide whether it is DDI and its type
#             pairs = s.getElementsByTagName("pair")
#             for p in pairs:
#                 try:
#                     if p.attributes["type"].value == label:
#                         print(stext)
#                         id_e1 = p.attributes["e1"].value
#                         id_e2 = p.attributes["e2"].value
#                         ddi_type = check_interaction(analysis, entities, id_e1, id_e2)
#                         break
#                 except KeyError:
#                     continue
