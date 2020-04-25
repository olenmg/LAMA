import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def compute_macro_avg_acc(args):
    """
    In LPAQA paper, macro-averaged accuracy computes accuracy for each unique object separately
    then averages the object/label averages together to get the relation-level accuracy
    """
    # List of unique object cluster average accuracies for all relations
    rel_acc_avgs_train = []
    rel_acc_avgs_dev = []
    rel_acc_avgs_test = []

    for subdir, dirs, files in os.walk(args.data_dir):
        rel = os.path.basename(subdir)
        # os.walk includes current directory as a subdir so skip it
        if not rel.startswith('P'):
            # This check could be more robust with a regex that makes sure the string is a 'P' followed by numbers
            continue

        obj_to_acc_train: Dict[Any, float] = defaultdict(float)
        obj_to_count_train: Dict[Any, int] = defaultdict(int)
        obj_to_acc_dev: Dict[Any, float] = defaultdict(float)
        obj_to_count_dev: Dict[Any, int] = defaultdict(int)
        obj_to_acc_test: Dict[Any, float] = defaultdict(float)
        obj_to_count_test: Dict[Any, int] = defaultdict(int)

        for f in files:
            filepath = os.path.join(subdir, f)
            with open(filepath, 'r') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    line = json.loads(line)
                    obj = line['obj'] #.lower()?
                    acc = line['acc']
                    if 'train' in f:
                        obj_to_acc_train[obj] += acc
                        obj_to_count_train[obj] += 1
                    elif 'dev' in f:
                        obj_to_acc_dev[obj] += acc
                        obj_to_count_dev[obj] += 1
                    elif 'test' in f:
                        obj_to_acc_test[obj] += acc
                        obj_to_count_test[obj] += 1

        # Unique object cluster
        obj_cluster_acc_avg_train = np.mean([obj_to_acc_train[k] / obj_to_count_train[k] for k in obj_to_acc_train])
        obj_cluster_acc_avg_dev = np.mean([obj_to_acc_dev[k] / obj_to_count_dev[k] for k in obj_to_acc_dev])
        obj_cluster_acc_avg_test = np.mean([obj_to_acc_test[k] / obj_to_count_test[k] for k in obj_to_acc_test])

        # Add accuracy average (of every unique object cluster) to list to later find relation-level accuracy
        rel_acc_avgs_train.append(obj_cluster_acc_avg_train)
        rel_acc_avgs_dev.append(obj_cluster_acc_avg_dev)
        rel_acc_avgs_test.append(obj_cluster_acc_avg_test)

    # MAA = Macro-Averaged Accuracy
    maa_train = np.mean(rel_acc_avgs_train)
    print('Train Macro-Averaged Accuracy:', round(maa_train * 100.0, 2))
    maa_dev = np.mean(rel_acc_avgs_dev)
    print('Dev Macro-Averaged Accuracy:', round(maa_dev * 100.0, 2))
    maa_test = np.mean(rel_acc_avgs_test)
    print('Test Macro-Averaged Accuracy:', round(maa_test * 100.0, 2))
    # abba = set([k.lower() for k in obj_to_acc_test.keys()])
    # print('lower cased set:', len(abba))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute macro-averaged accuracy for train, dev, test sets of each TREx relation (i.e. macro directory)')
    parser.add_argument('data_dir', type=str, help='Directory containing object accuracies for each relation')
    args = parser.parse_args()

    compute_macro_avg_acc(args)
