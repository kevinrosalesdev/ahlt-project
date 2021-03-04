import sys


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


if __name__ == '__main__':
    model_name = sys.argv[1]
    features = sys.argv[2]
    pass