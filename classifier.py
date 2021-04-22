import sys

if __name__ == '__main__':
    # read each vector in input file
    for line in sys.stdin:
        # split line into elements
        fields = line.strip('\n').split('\t')
        # first 4 elements are sid, e1, e2 and ground truth (ignored since we are classifying)
        (sid, e1, e2, gt) = fields[0:4]

        # rest of elements are features, passed to the classifier of choice to get a prediction
        # prediction = mymodel.classify(fields[4:])
        prediction = 'null'

        # if the classifier predicted a DDI, output it in the right format
        if prediction != 'null':
            print(sid, e1, e2, prediction, sep='|')
