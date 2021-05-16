# extract features for train and devel datasets
#python3 feature_extractor.py data/train/ > train_summary.feat
#python3 feature_extractor.py data/devel/ > devel_summary.feat
# use train dataset to learn a model
python3 learner.py mymodel_summary train_summary.feat mlb_summary
# annotate devel dataset using learned model
python3 classifier.py mymodel_summary devel_summary.feat mlb_summary > devel_summary.out
# evaluate performance of the model
python3.7 evaluator.pyc DDI data/devel/ devel_summary.out