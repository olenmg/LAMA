# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
from operator import itemgetter
import random
import numpy as np

# [CONFIGURABLE]: uncomment BERT settings if you want to evaluate with BERT and vice versa with RoBERTa
LMs = [
    # {
    #     "lm": "bert",
    #     "label": "bert_base",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-base-cased",
    #     "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12"
    # },
    {
        "lm": "roberta",
        "label": "roberta_large",
        "models_names": ["roberta"],
        "roberta_model_name": "model.pt",
        "roberta_model_dir": "pre-trained_language_models/roberta/roberta.large",
        "roberta_vocab_name": "dict.txt"
    }
]

def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    metrics,
    dataset_type,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    all_MRR = []
    all_Precision = []
    all_Precision1_RE = []

    results_file = open("last_results.csv", "w+")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            # Google RE and TREx
            "dataset_filename": "{}/{}/{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_cased_rob.txt", # [CONFIGURABLE]: BERT -> common_vocab_cased.txt,  BERT + RoBERTa -> common_vocab_cased_rob.txt
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 64, # [CONFIGURABLE]: 64 for Fact Retrieval and 32 for Relation Extraction (on a NVIDIA GeForce GTX 1080ti)
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 50, # used to be 100
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
            "use_ctx": False, # [CONFIGURABLE]: Toggle for Relation Extraction
            "synthetic": False # [CONFIGURABLE]: Toggle for perturbed sentence evaluation for Relation Extraction
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # This is for easy recording of metrics across train, dev, test sets for each relation
        if relation['relation'] not in metrics:
            metrics[relation['relation']] = {
                'train': {
                    'mrr': 0,
                    'p10': 0,
                    'p1': 0
                },
                'dev': {
                    'mrr': 0,
                    'p10': 0,
                    'p1': 0
                },
                'test': {
                    'mrr': 0,
                    'p10': 0,
                    'p1': 0
                }
            }

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)
        MRR, Precision, Precision1, Precision1_RE = run_evaluation(args, relation['relation'], shuffle_data=False, model=model, use_ctx=args.use_ctx, synthetic=args.synthetic)
        print("P@1 : {}".format(Precision1), flush=True)
        all_MRR.append(MRR)
        all_Precision.append(Precision)
        all_Precision1.append(Precision1)
        all_Precision1_RE.append(Precision1_RE)

        # This is for easy recording of metrics across train, dev, test sets for each relation
        if args.use_ctx:
            metrics[relation['relation']][dataset_type]['p1'] = round(Precision1_RE * 100.0, 2)
        else:
            metrics[relation['relation']][dataset_type]['mrr'] = round(MRR * 100.0, 2)
            metrics[relation['relation']][dataset_type]['p10'] = round(Precision * 100.0, 2)
            metrics[relation['relation']][dataset_type]['p1'] = round(Precision1 * 100.0, 2)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    if args.use_ctx:
        mean_p1_re = np.mean(all_Precision1_RE)
        print("MEAN P@1 RE: {}".format(mean_p1_re))
    else:
        mean_mrr = np.mean(all_MRR)
        print('MEAN MRR: {}'.format(mean_mrr))
        mean_p10 = np.mean(all_Precision)
        print('MEAN P@10: {}'.format(mean_p10))

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_post, data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    ############################################ [CONFIGURABLE] ############################################
    """
    For fact retrieval, the data path would look something like "../data/fact_retrieval/original_rob"
    """
    data_path_pre = "../data/relation_extraction"
    ########################################################################################################
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters, metrics, dataset_type):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, metrics, dataset_type, input_param=ip, use_negated_probes=False)


def print_all_relation_metrics(metrics):
    avg_mrr_test = 0
    avg_p10_test = 0
    avg_p1_test = 0
    for relation in metrics:
        rel_train = metrics[relation]['train']
        rel_dev = metrics[relation]['dev']
        rel_test = metrics[relation]['test']
        # Calculate averages
        avg_mrr_test += rel_test['mrr']
        avg_p10_test += rel_test['p10']
        avg_p1_test += rel_test['p1']
        print('{}: {} & {} & {} & {} & {} & {} & {} & {} & {}'.format(relation,
                                                                    rel_train['mrr'],
                                                                    rel_train['p10'],
                                                                    rel_train['p1'],
                                                                    rel_dev['mrr'],
                                                                    rel_dev['p10'],
                                                                    rel_dev['p1'],
                                                                    rel_test['mrr'],
                                                                    rel_test['p10'],
                                                                    rel_test['p1']))
    print('=' * 80)
    print('Average Test MRR:', avg_mrr_test / len(metrics))
    print('Average Test P@10:', avg_p10_test / len(metrics))
    print('Average Test P@1:', avg_p1_test / len(metrics))


if __name__ == "__main__":
    metrics = {}
    print(('='*40) + ' TRAIN ' + ('='*40))
    data_path_post = 'train.jsonl'
    parameters = get_TREx_parameters(data_path_post)
    run_all_LMs(parameters, metrics, 'train')
    print(('='*40) + ' DEV ' + ('='*40))
    data_path_post = 'dev.jsonl'
    parameters = get_TREx_parameters(data_path_post)
    run_all_LMs(parameters, metrics, 'dev')
    print(('='*40) + ' TEST ' + ('='*40))
    data_path_post = 'test.jsonl'
    parameters = get_TREx_parameters(data_path_post)
    run_all_LMs(parameters, metrics, 'test')
    print(('='*40) + ' METRICS ' + ('='*40))
    print_all_relation_metrics(metrics)
