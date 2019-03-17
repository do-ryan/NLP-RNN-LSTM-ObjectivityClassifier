"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(file_path):
    tsv_file = pd.read_csv(file_path, sep='\t')
    tsv_file.reset_index(drop=True)

    objective_samples = tsv_file[tsv_file['label']==0]
    subjective_samples = tsv_file[tsv_file['label']==1]
    # split into objective/subjective

    obj_train = objective_samples.sample(frac = 0.64, random_state=1)
    obj_val = objective_samples.drop(obj_train.index).sample(frac = 0.16/(1-0.64), random_state=1)
    obj_test = objective_samples.drop(obj_train.index).drop(obj_val.index)
    # split objective into train/val/test

    subj_train = subjective_samples.sample(frac=0.64, random_state=1)
    subj_val = subjective_samples.drop(subj_train.index).sample(frac=0.16 / (1 - 0.64), random_state=1)
    subj_test = subjective_samples.drop(subj_train.index).drop(subj_val.index)
    # split subjective into train/val/test

    train = pd.concat([obj_train, subj_train])
    val = pd.concat([obj_val, subj_val])
    test = pd.concat([obj_test, subj_test])

    print ("Num objective training examples: {}. Num subjective training examples: {}".format(len(obj_train), len(subj_train)))
    print("Num objective validation examples: {}. Num subjective validation examples: {}".format(len(obj_val), len(subj_val)))
    print("Num objective test examples: {}. Num subjective test examples: {}".format(len(obj_test), len(subj_test)))

    train.to_csv('./data/train.tsv', sep='\t', index=False, header=False)
    val.to_csv('./data/val.tsv', sep='\t', index=False, header=False)
    test.to_csv('./data/test.tsv', sep='\t', index=False, header=False)

    return

if __name__ == '__main__':

    train_val_test_split("./data/data.tsv")
