import sys
import networkx as nx

from os import listdir
from xml.dom.minidom import parse

import numpy as np
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
        print(offsets)
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

    return idx_e[0]


def write_direct_path(analysis, idx_1, idx_2, mode='default'):
    """
    :param mode:
        - 'default': Slides 21-23 (e.g. path=conj<dobj)
        - 'entities': Slide 24 (e.g. path=conj<ENTITY/dobj)
        - 'summary': Slide 25 (e.g. path=dobj*)
    """
    if mode == 'entities':
        entities_idx = [get_entity_idx(analysis, entities, e) for e in list(entities.keys())]

    try:
        direct_path = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_1, idx_2)[:-1][::-1]
        if mode == 'summary' and len(direct_path) > 1:
            return f"{analysis.nodes[direct_path[0]]['rel']}*<"
        path = ''
        for i in direct_path:
            if mode == 'entities' and i in entities_idx:
                path = f"ENTITY/{analysis.nodes[i]['rel']}<{path}"
            else:
                path = f"{analysis.nodes[i]['rel']}<{path}"
        return path
    except nx.exception.NetworkXNoPath:
        pass

    try:
        direct_path = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_2, idx_1)[:-1][::-1]
        if mode == 'summary' and len(direct_path) > 1:
            return f">{analysis.nodes[direct_path[0]]['rel']}*"
        path = ''
        for i in direct_path:
            if mode == 'entities' and i in entities_idx:
                path = f"{path}>ENTITY/{analysis.nodes[i]['rel']}"
            else:
                path = f"{path}>{analysis.nodes[i]['rel']}"
        return path
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
    if mode != 'summary':
        path = analysis.nodes[idx]['lemma']
    else:
        path = ''
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


def extract_features(tree: DependencyGraph, entities, e1, e2, mode='default'):
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
    open_class = ['JJ', 'JJR', 'JJS'
                               'NN', 'NNS', 'NNP', 'NNPS',
                  'RB', 'RBR', 'RBS',
                  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    for idx in range(1, idx_1):
        if tree.nodes[idx]['tag'] in open_class:
            feature_list.append(f"lb1={tree.nodes[idx]['lemma']}")

    for idx in range(idx_1 + 1, idx_2):
        if tree.nodes[idx]['tag'] in open_class:
            feature_list.append(f"lib={tree.nodes[idx]['lemma']}")

    for idx in range(idx_2 + 1, len(tree.nodes)):
        if tree.nodes[idx]['tag'] in open_class:
            feature_list.append(f"la2={tree.nodes[idx]['lemma']}")

    entity_in_between = False
    offset_1 = entities[e1]
    offset_2 = entities[e2]
    for entity in entities.keys():
        offset_3 = entities[entity]
        if entity != e1 and entity != e2 and int(offset_3[0]) > int(offset_1[-1]) and int(offset_3[-1]) < int(
                offset_2[0]):
            entity_in_between = True
            break

    if entity_in_between:
        feature_list.append('entity_in_between')

    direct_path = write_direct_path(tree, idx_1, idx_2, mode=mode)
    if direct_path is not None:
        feature_list.append(f'path={direct_path}')
    else:
        effects_list = ['administer', 'potentiate', 'prevent', 'block', 'cause', 'enhance', 'result', 'associate']
        mech_list = ['reduce', 'increase', 'decrease']
        int_list = ['interact', 'interaction']
        advise_list = ['advise', 'recommend', 'caution', 'consider']
        total_list = [effects_list, mech_list, int_list, advise_list]

        lcs = np.Inf
        lcs_path = None
        # verb_info = [None, None, None]
        for idx in range(1, len(tree.nodes)):
            # It only returns the path containing the LCS Verb!!
            if tree.nodes[idx]['tag'] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                path = write_path(tree, idx_1, idx_2, idx, entities, mode=mode)
                should = False
                try:
                    auxiliaries = tree.nodes[idx]['deps']['aux']
                    for node in auxiliaries:
                        if tree.nodes[node]['lemma'].lower() == 'should':
                            should = True
                            break
                except KeyError:
                    pass
                search_list = [i for i in range(len(total_list)) if tree.nodes[idx]['lemma'] in total_list[i]]
                if path is not None and path.count("<") + path.count(">") < lcs:
                    lcs_path = path
                    lcs = path.count("<") + path.count(">")
                    verb_info = [tree.nodes[idx]['tag'], tree.nodes[idx]['lemma'],
                                 search_list[0] if len(search_list) != 0 else None,
                                 should, path]
                elif path is not None:
                    feature_list.append(f'path={path}')
                    feature_list.append(f"pos={tree.nodes[idx]['tag']}")
                    feature_list.append(f"lemma={tree.nodes[idx]['lemma']}")
                    search = search_list[0] if len(search_list) != 0 else None
                    if search is not None:
                        type_list = ['effect', 'mech', 'int', 'advise']
                        feature_list.append(f'list={type_list[search]}')
                    if should:
                        feature_list.append(f'should')

        if lcs_path is not None:
            feature_list.append(f'LCSpath={lcs_path}')
            feature_list.append(f'LCSpos={verb_info[0]}')
            feature_list.append(f'LCSlemma={verb_info[1]}')
            if verb_info[2] is not None:
                type_list = ['effect', 'mech', 'int', 'advise']
                feature_list.append(f'LCSlist={type_list[verb_info[2]]}')
            if verb_info[3]:
                feature_list.append(f'LCSshould')

    return feature_list


if __name__ == '__main__':
    inputdir = sys.argv[1]
    mode = sys.argv[2]

    # connect to your CoreNLP server (just once)
    my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

    # process each file in directory
    for idx, f in enumerate(listdir(inputdir), 1):
        # print(f"Processing file nº {idx}/{len(listdir(inputdir))}")

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
                feats = extract_features(analysis, entities, id_e1, id_e2, mode=mode)
                print(sid, id_e1, id_e2, dditype, '\t'.join(feats), sep="\t")

# should+paths
#                 tp     fp      fn    #pred   #exp    P       R       F1
# ------------------------------------------------------------------------------
# advise             82     62      56     144     138    56.9%   59.4%   58.2%
# effect            126     78     189     204     315    61.8%   40.0%   48.6%
# int                16      2      19      18      35    88.9%   45.7%   60.4%
# mechanism         101     79     163     180     264    56.1%   38.3%   45.5%
# ------------------------------------------------------------------------------
# M.avg            -      -       -       -       -       65.9%   45.8%   53.1%
# ------------------------------------------------------------------------------
# m.avg             325    221     427     546     752    59.5%   43.2%   50.1%
# m.avg(no class)   352    194     400     546     752    64.5%   46.8%   54.2%

# should
#                 tp     fp      fn    #pred   #exp    P       R       F1
# ------------------------------------------------------------------------------
# advise             76     62      62     138     138    55.1%   55.1%   55.1%
# effect            123     75     192     198     315    62.1%   39.0%   48.0%
# int                16      2      19      18      35    88.9%   45.7%   60.4%
# mechanism          97     86     167     183     264    53.0%   36.7%   43.4%
# ------------------------------------------------------------------------------
# M.avg            -      -       -       -       -       64.8%   44.1%   51.7%
# ------------------------------------------------------------------------------
# m.avg             312    225     440     537     752    58.1%   41.5%   48.4%
# m.avg(no class)   343    194     409     537     752    63.9%   45.6%   53.2%

# path
#                 tp     fp      fn    #pred   #exp    P       R       F1
# ------------------------------------------------------------------------------
# advise             76     62      62     138     138    55.1%   55.1%   55.1%
# effect            123     75     192     198     315    62.1%   39.0%   48.0%
# int                16      2      19      18      35    88.9%   45.7%   60.4%
# mechanism          97     86     167     183     264    53.0%   36.7%   43.4%
# ------------------------------------------------------------------------------
# M.avg            -      -       -       -       -       64.8%   44.1%   51.7%
# ------------------------------------------------------------------------------
# m.avg             312    225     440     537     752    58.1%   41.5%   48.4%
# m.avg(no class)   343    194     409     537     752    63.9%   45.6%   53.2%

# None
#                 tp     fp      fn    #pred   #exp    P       R       F1
# ------------------------------------------------------------------------------
# advise             73     59      65     132     138    55.3%   52.9%   54.1%
# effect            128     68     187     196     315    65.3%   40.6%   50.1%
# int                16      2      19      18      35    88.9%   45.7%   60.4%
# mechanism          98     76     166     174     264    56.3%   37.1%   44.7%
# ------------------------------------------------------------------------------
# M.avg            -      -       -       -       -       66.5%   44.1%   52.3%
# ------------------------------------------------------------------------------
# m.avg             315    205     437     520     752    60.6%   41.9%   49.5%
# m.avg(no class)   346    174     406     520     752    66.5%   46.0%   54.4%
