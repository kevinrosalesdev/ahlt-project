import sys, pickle
import tensorflow as tf
import numpy as np

from learner_DDI import load_data, encode_words


def load_model_and_indexs(filename):
    """
    Task: Load model and associate indexes from disk.
    Input:
        - filename: filename to be loaded.
    Output:
        Loads a model from filename.nn and its indexes from filename.idx.
        Returns the loaded model and indexes.
    """
    model = tf.keras.models.load_model(f"{filename}.nn.h5")
    file = open(f"{filename}.idx", "rb")
    idx = pickle.load(file)
    file.close()

    return model, idx


def output_interactions(dataset, preds):
    """
    Task: Output detected DDIs in the format expected by the evaluator.

    Input:
        dataset: A dataset produced by load_data.
        preds: For each sentence in dataset, a label for its DDI type (or
               'null' if no DDI detected)

    Output:
        Prints the detected interactions to sdout in the format required by the
        evaluator.

    Example:
        >> output_interactions(dataset, preds)
        DDI-DrugBank.d398.s0|DDI-DrugBank.d398.s0.e0|DDI-DrugBank.d398.s0.e1|effect
        DDI-DrugBank.d398.s0|DDI-DrugBank.d398.s0.e0|DDI-DrugBank.d398.s0.e2|effect
        DDI-DrugBank.d398.s2|DDI-DrugBank.d398.s2.e0|DDI-DrugBank.d398.s2.e5|mechanism
        ...
    """

    # if the classifier predicted a DDI, output it in the right format
    for idx, prediction in enumerate(preds):
        if prediction != 'null':
            print(dataset[idx][0], dataset[idx][1], dataset[idx][2], prediction, sep='|')


def predict(modelname, datadir):
    """
    Loads a NN Model from file 'modelname' and uses it to extract drugs
    in datadir. Saves results to 'outfile' in the appropriate format.
    """

    # load model and associated encoding data
    model, idx = load_model_and_indexs(modelname)

    # load data to annotate
    test_data = load_data(datadir, load_pickle='val')

    # encode dataset
    X = encode_words(test_data, idx)

    # tag sentences in dataset
    Y = model.predict(X)

    keys = list(idx['labels'].keys())
    values = list(idx['labels'].values())
    # get most likely tag for each word
    Y = [keys[values.index(np.argmax(y))] for y in Y]

    # extract entities and dump them to output file
    output_interactions(test_data, Y)

    # evaluate using official evaluator
    # evaluator.evaluate("NER", datadir, outfile)


if __name__ == '__main__':

    # This code may be needed for some GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    modelname = sys.argv[1]
    datadir = sys.argv[2]
    predict(modelname, datadir)
