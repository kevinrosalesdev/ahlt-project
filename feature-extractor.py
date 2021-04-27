import sys
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
    # '\r\n': Symbols found in training set that do not allow CoreNLP to correctly process the sentence.
    s = s.replace("%", "%25").replace("+", "%2B").replace("\r\n", ". ")
    if len(s) > 0:
        # parse text (as many times as needed).
        mytree, = parser.raw_parse(s)  # type: DependencyGraph
        offsets = parser.api_call(s, properties={'annotators': 'tokenize'})['tokens']
        for idx in range(len(mytree.nodes) - 1):
            mytree.nodes[idx + 1]['start'] = offsets[idx]['characterOffsetBegin']
            mytree.nodes[idx + 1]['end'] = offsets[idx]['characterOffsetEnd'] - 1
        return mytree
    else:
        print("Empty sentence!")
        return {}


def get_entity_idx(tree: DependencyGraph, entities, e):
    offset = entities[e]

    # Some GT offsets are ""215-226;246-276" because the drug is not consecutive
    if len(offset) > 2:
        offset = [offset[0], offset[-1]]

    idx_e = [0, 10]

    for idx in range(1, len(tree.nodes)):
        for shift in range(0, 10):
            if idx_e[1] > shift and tree.nodes[idx]['start'] >= int(offset[0]) - shift \
                    and tree.nodes[idx]['end'] <= int(offset[1]) + shift:
                idx_e = [idx, shift]

    # Rules
    return idx_e[0]


def write_direct_path(analysis, idx_1, idx_2, mode='default'):
    """
    :param mode:
        - 'default': Slides 21-23 (e.g. path=conj<dobj<combine>nmod)
        - 'entities': Slide 24 (e.g. path=conj<ENTITY/dobj<combine>nmod)
        - 'summary': Slide 25 (e.g. path=dobj*<combine>nmod)
    """
    if mode == 'entities':
        entities_idx = [get_entity_idx(analysis, entities, e) for e in list(entities.keys())]

    try:
        direct_path = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_1, idx_2)[:-1][::-1]
        if mode == 'summary' and len(direct_path) > 1:
            return f"{analysis.nodes[direct_path[0]]['rel']}*"
        path = ''
        for i in direct_path:
            if mode == 'entities' and i in entities_idx:
                path = f"ENTITY/{analysis.nodes[i]['rel']}<{path}"
            else:
                path = f"{analysis.nodes[i]['rel']}<{path}"
        return path[:-1]
    except nx.exception.NetworkXNoPath:
        pass

    try:
        direct_path = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_2, idx_1)[:-1][::-1]
        if mode == 'summary' and len(direct_path) > 1:
            return f"{analysis.nodes[direct_path[0]]['rel']}*"
        path = ''
        for i in direct_path:
            if mode == 'entities' and i in entities_idx:
                path = f"{path}>ENTITY/{analysis.nodes[i]['rel']}"
            else:
                path = f"{path}>{analysis.nodes[i]['rel']}"
        return path[1:]
    except nx.exception.NetworkXNoPath:
        pass

    return None


def write_path(analysis, idx_1, idx_2, idx, entities, mode='default'):
    """
    :param mode:
        - 'default': Slides 21-23 (e.g. path=conj<dobj<combine>nmod)
        - 'entities': Slide 24 (e.g. path=conj<ENTITY/dobj<combine>nmod)
        - 'summary': Slide 25 (e.g. path=dobj*<combine>nmod)
    """
    if mode == 'entities':
        entities_idx = [get_entity_idx(analysis, entities, e) for e in list(entities.keys())]

    try:
        path1 = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_1, idx)[:-1][::-1]
        path2 = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_2, idx)[:-1][::-1]
    except nx.exception.NetworkXNoPath:
        return None
    path = analysis.nodes[idx]['lemma']
    for i in path1:
        if mode == 'summary' and len(path1) > 1:
            path = f"{analysis.nodes[path1[0]]['rel']}*<{path}"
            break
        if mode == 'entities' and i in entities_idx:
            path = f"ENTITY/{analysis.nodes[i]['rel']}<{path}"
        else:
            path = f"{analysis.nodes[i]['rel']}<{path}"
    for i in path2:
        if mode == 'summary' and len(path2) > 1:
            path = f"{path}>{analysis.nodes[path2[0]]['rel']}*"
            break
        if mode == 'entities' and i in entities_idx:
            path = f"{path}>ENTITY/{analysis.nodes[i]['rel']}"
        else:
            path = f"{path}>{analysis.nodes[i]['rel']}"

    return path


def extract_features(tree: DependencyGraph, entities, e1, e2):
    """
    Task:
        Given an analyzed sentence and two target entities, compute a feature
        vector for this classification example.
    Input:
        tree: a DependencyGraph object with all sentence information.
        entities: A list of all entities in the sentence (id and offsets).
        e1, e2: ids of the two entities to be checked for an interaction
    Output:
        A vector of binary features.
        Features are binary and vectors are in sparse representation (i.e. only
        active features are listed)
    Example:
        >> extract_features (tree, {’DDI - DrugBank. d370.s1.e0’:[’43’,’52’],
                                    ’DDI - DrugBank. d370.s1.e1’:[’57’,’70’],
                                    ’DDI - DrugBank. d370.s1.e2’:[’77’,’88’]},
                             ’DDI - DrugBank. d370.s1.e0’,’DDI - DrugBank. d370.s1.e2’)
           [’lb1=Caution’,’lb1=be’,’lb1=exercise’,’lb1=combine’,’lib=or’,’lib=salicylic’,
           ’lib=acid’,’lib=with’,’LCSpos=VBG’,’LCSlema=combine’,
            ’path=dobj/combine\nmod\compound’ ’entity_in_between’]
    """
    idx_1 = get_entity_idx(tree, entities, e1)
    idx_2 = get_entity_idx(tree, entities, e2)

    feature_list = []
    direct_path = write_direct_path(tree, idx_1, idx_2, mode="default")
    if direct_path is not None:
        feature_list.append(f'dir_path={direct_path}')
    for idx in range(1, len(analysis.nodes)):
        if analysis.nodes[idx]['tag'] in ['VB', 'VBN', 'VBZ', 'VBG']:
            path = write_path(tree, idx_1, idx_2, idx, entities, mode="default")
            if path is not None:
                feature_list.append(f'path={path}')

    return feature_list


if __name__ == '__main__':
    inputdir = sys.argv[1]

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

            # analyze sentence if there is at least a pair of entities
            if len(entities) > 1:
                analysis = analyze(stext, my_parser)

            # for each pair in the sentence, decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                # get ground truth
                ddi = p.attributes["ddi"].value
                dditype = p.attributes["type"].value if ddi == 'true' else 'null'

                # target entities
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value

                # feature extraction
                feats = extract_features(analysis, entities, id_e1, id_e2)
                print(sid, id_e1, id_e2, dditype, '\t'.join(feats), sep="\t")
