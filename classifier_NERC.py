import tensorflow.keras.models
import pickle
import numpy as np
import sys

from learner_NERC import load_data, encode_words


def load_model_and_indexs(filename):    # Kevin
    """
    Task: Load model and associate indexes from disk.
    Input:
        - filename: filename to be loaded.
    Output:
        Loads a model from filename.nn and its indexes from filename.idx.
        Returns the loaded model and indexes.
    """
    model = tensorflow.keras.models.load_model(f"{filename}.nn.h5")
    file = open(f"{filename}.idx", "rb")
    idx = pickle.load(file)
    file.close()

    return model, idx


def output_entities(words, preds, offsets, sids, max_length):  # Kevin
    """
    Task:
        Output detected entities in the format expected by the evaluator.
    Input:
        dataset -> (words, offsets, sids): A dataset produced by load_data
        preds: For each sentence in the dataset, a list with the labels for
               each sentence token, as predicted by the model.
        outfile: Output file.
    Output:
        Prints the detected entities to stdout in the format required by the evaluator.
    Example:
        >> output_entities(dataset, preds):
        DDI-DrugBank.d283.s4|14-35|bile acid sequestrants|group
        DDI-DrugBank.d283.s4|99-104|tricor|group
        DDI-DrugBank.d283.s5|22-23|cyclosporine|drug
        DDI-DrugBank.d283.s5|196-208|fibrate drugs|group
        ...
    """
    for sentence in range(len(sids)):
        last_found = 'O'
        for index in range(min(len(words[sentence]), max_length)):
            if preds[sentence][index][0] == 'B':
                # B-tag & the last tag was different to O-tag -> New entity must be printed.
                if last_found != 'O':
                    print(f"{sids[sentence]}|{offset[0]}-{offset[1]}|{name}|{type}")

                name = words[sentence][index][0]
                name = name[name.find("=")+1:]
                offset = [offsets[sentence][index][0], offsets[sentence][index][1]]
                type = preds[sentence][index][2:]
                last_found = 'B'

            elif preds[sentence][index][0] == 'I':
                try:
                    word = words[sentence][index][0]
                    name += " " + word[word.find("=")+1:]
                    offset[1] = offsets[sentence][index][1]
                    last_found = 'I'
                except:
                    continue

            else:
                # O-tag & the last tag was different to O-tag -> New entity must be printed.
                if last_found != 'O':
                    print(f"{sids[sentence]}|{offset[0]}-{offset[1]}|{name}|{type}")
                last_found = 'O'

        # If the last tag of the array was different from O, it must be printed too.
        if last_found != 'O':
            print(f"{sids[sentence]}|{offset[0]}-{offset[1]}|{name}|{type}")


def predict(modelname, datadir):  # Kevin
    """
    Loads a NN Model from file 'modelname' and uses it to extract drugs
    in datadir. Saves results to 'outfile' in the appropriate format.
    """

    # load model and associated encoding data
    model, idx = load_model_and_indexs(modelname)

    # load data to annotate
    words_test, tags_test, sids_test, gt_test, offsets_test = load_data(datadir)

    # encode dataset
    X = np.array(encode_words(words_test, idx)['form'])

    # tag sentences in dataset
    Y = model.predict(X)

    keys = list(idx['labels'].keys())
    values = list(idx['labels'].values())

    # get most likely tag for each word
    Y = [[keys[values.index(np.argmax(y))] for y in s] for s in Y]

    # extract entities and dump them in output file
    output_entities(words_test, Y, offsets_test, sids_test, max_length=idx['max_length'])

    # evaluate using official evaluator
    # evaluator.evaluate("NER", datadir, outfile)


if __name__ == '__main__':
    modelname = sys.argv[1]
    datadir = sys.argv[2]
    predict(modelname, datadir)
