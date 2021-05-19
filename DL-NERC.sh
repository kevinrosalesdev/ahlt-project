# Use train dataset to learn a model,
# use devel dataset for validation.
# python3.7 learner_NERC.py data/train data/devel model
# annotate devel dataset using learned model
python3.7 classifier_NERC.py model data/devel > devel.out
# evaluate performance of the model
python3.7 evaluator.pyc NER data/devel/ devel.out