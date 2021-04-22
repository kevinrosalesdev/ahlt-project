# extract features for train and devel datasets
python3 feature-extractor.py data/train/ > train.feat
python3 feature-extractor.py data/devel/ > devel.feat
# use train dataset to learn a model
python3 learner.py mymodel train.feat
# annotate devel dataset using learned model
python3 classifier.py mymodel devel.feat > devel.out
# evaluate performance of the model
python3 evaluator.pyc DDI data/devel/ devel.out