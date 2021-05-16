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


# Gets the path between two indexes of the Dependency Graph with respect to another index (e.g. a verb)
def write_path(analysis, idx_1, idx_2, idx):
    path1 = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_1, idx)
    path2 = nx.astar_path(nx.DiGraph(analysis.nx_graph()), idx_2, idx)
    path = analysis.nodes[idx]['word']
    # Keep information about the words and the relations in the path
    for i in path1[:-1][::-1]:
        path = f"{analysis.nodes[i]['word']} [{analysis.nodes[i]['rel']}] > {path}"
    for i in path2[:-1][::-1]:
        path = f"{path} < [{analysis.nodes[i]['rel']}] {analysis.nodes[i]['word']}"

    path1, path2 = path.split(f"> {analysis.nodes[idx]['word']} <")
    return path, path1, path2


# Returns the verb type (considering lists for each type of verb)
def get_list(lemma, lists):
    if lemma in lists[3]:
        return 'advise'
    elif lemma in lists[0]:
        return 'effect'
    elif lemma in lists[1]:
        return 'mechanism'
    elif lemma in lists[2]:
        return 'int'
    return False


def check_interaction(analysis: DependencyGraph, entities, e1, e2):
    offset_1 = entities[e1]
    offset_2 = entities[e2]
    # Some GT offsets are like "215-226;246-276" because the drug is not consecutive, so
    # only offset[0] and offset[-1] is considered (e.g. "215-276").
    if len(offset_1) > 2:
        offset_1 = [offset_1[0], offset_1[-1]]
    if len(offset_2) > 2:
        offset_2 = [offset_2[0], offset_2[-1]]

    # Some CoreNLP offsets do not match the GT, so a small shift must be performed to find them.
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

    idx_1 = idx_1[0]
    idx_2 = idx_2[0]

    # Rules

    effects_list = ['administer', 'potentiate', 'prevent', 'block', 'cause', 'enhance', 'result', 'associate']
    mech_list = ['reduce', 'increase', 'decrease']
    int_list = ['interact', 'interaction']
    advise_list = ['advise', 'recommend', 'caution', 'consider']

    total_list = [effects_list, mech_list, int_list, advise_list]

    # If both drugs are the same, there is no interaction.
    if analysis.nodes[idx_1]['word'].lower() == analysis.nodes[idx_2]['word'].lower():
        return None

    # If the lemma from a token is 'should', take its head.
    # If both drugs can write a path to this head, the interaction is 'advise'.
    for idx in range(1, len(analysis.nodes)):
        if analysis.nodes[idx]['lemma'].lower() == 'should':
            try:
                head = analysis.nodes[idx]['head']
                write_path(analysis, idx_1, idx_2, head)
                return 'advise'
            except nx.exception.NetworkXNoPath:
                continue

    # If the lemma from a token is inside some verb list, write the path from both drugs to this verb.
    # If the path includes the [nsubj] in the first part of the path (before the verb) and the [obj]
    # in the second one (after the verb) or the other way around, an interaction is returned
    # (its type depends on the type of verb considering the lists).
    for idx in range(1, len(analysis.nodes)):
        lemma_list = get_list(analysis.nodes[idx]['lemma'].lower(), total_list)
        try:
            if lemma_list:
                path, path1, path2 = write_path(analysis, idx_1, idx_2, idx)
                if ('[nsubj]' in path1 and '[obj]' in path2) or ('[nsubj]' in path2 and '[obj]' in path1):
                    return lemma_list
        except nx.exception.NetworkXNoPath:
            continue

    # Similar to previous rule but less strict since it does not require any [nsubj] or [obj] relations.
    for idx in range(1, len(analysis.nodes)):
        lemma_list = get_list(analysis.nodes[idx]['lemma'].lower(), total_list)
        try:
            if lemma_list:
                write_path(analysis, idx_1, idx_2, idx)
                return lemma_list
        except nx.exception.NetworkXNoPath:
            continue

    return None

"""
TEST:
                    tp	  fp	  fn	#pred	#exp	    P	    R	    F1
------------------------------------------------------------------------------
advise            150	 416	  62	 566	 212	26.5%	70.8%	38.6%
effect             74	 370	 209	 444	 283	16.7%	26.1%	20.4%
int                 7	 172	  11	 179	  18	3.9%	38.9%	7.1%
mechanism         110	 578	 227	 688	 337	16.0%	32.6%	21.5%
------------------------------------------------------------------------------
M.avg               -	    -	    -	    -	    -	15.8%	42.1%	21.9%
------------------------------------------------------------------------------
m.avg             341	1536	 509	1877	 850	18.2%	40.1%	25.0%
m.avg(no class)   470	1407	 380	1877	 850	25.0%	55.3%	34.5%

DEVEL:
                   tp	  fp	  fn	#pred	#exp	P	R	F1
------------------------------------------------------------------------------
advise             96	 168	  42	 264	 138	36.4%	69.6%	47.8%
effect             66	 342	 249	 408	 315	16.2%	21.0%	18.3%
int                32	 232	   3	 264	  35	12.1%	91.4%	21.4%
mechanism          60	 347	 204	 407	 264	14.7%	22.7%	17.9%
------------------------------------------------------------------------------
M.avg               -	    -	    -	    -	    -	19.9%	51.2%	26.3%
------------------------------------------------------------------------------
m.avg             254	1089	 498	1343	 752	18.9%	33.8%	24.2%
m.avg(no class)   374	 969	 378	1343	 752	27.8%	49.7%	35.7%


TRAIN:
                    tp	  fp	  fn	#pred	#exp	P	R	F1
------------------------------------------------------------------------------
advise            480	1146	 217	1626	 697	29.5%	68.9%	41.3%
effect            334	1343	1116	1677	1450	19.9%	23.0%	21.4%
int               153	2441	  78	2594	 231	5.9%	66.2%	10.8%
mechanism         245	2989	 775	3234	1020	7.6%	24.0%	11.5%
------------------------------------------------------------------------------
M.avg                -	   -	   -	   -	   -	15.7%	45.5%	21.3%
------------------------------------------------------------------------------
m.avg            1212	7919	2186	9131	3398	13.3%	35.7%	19.3%
m.avg(no class)  1768	7363	1630	9131	3398	19.4%	52.0%	28.2%
"""

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