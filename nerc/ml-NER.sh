#!/bin/bash

python3 feature-extractor.py data/train/ > train.feat
python3 feature-extractor.py data/devel/ > devel.feat

python3 learner.py crf train.feat

python3 classifier.py crf devel.feat > result.out

python3 eval/evaluator.py NER data/devel/ result.out