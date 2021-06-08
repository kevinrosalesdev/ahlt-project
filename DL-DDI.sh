# Use train dataset to learn a model,
# use devel dataset for validation.
# python3 learner_DDI.py data/train data/devel model-ddi
# annotate devel dataset using learned model
python3 classifier_DDI.py model-ddi data/devel > devel.out
# evaluate performance of the model
python3.7 evaluator.pyc DDI data/devel devel.out