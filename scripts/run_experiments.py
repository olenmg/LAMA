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

LMs = [
    # {
    #     "lm": "bert",
    #     "label": "bert_base",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-base-cased",
    #     "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12"
    # },
    # {
    #     "lm": "bert",
    #     "label": "bert_large",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-large-cased",
    #     "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    # }
    {
        "lm": "roberta",
        "label": "roberta_base",
        "models_names": ["roberta"],
        "roberta_model_name": "model.pt",
        "roberta_model_dir": "pre-trained_language_models/roberta/roberta.base",
        "roberta_vocab_name": "dict.txt"
    }
]

LM_ROBERTA_BASE = {
    "lm": "roberta",
    "label": "roberta_base",
    "models_names": ["roberta"],
    "roberta_model_name": "model.pt",
    "roberta_model_dir": "pre-trained_language_models/roberta/roberta.base",
    "roberta_vocab_name": "dict.txt"
}

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
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            # Google RE and TREx
            "dataset_filename": "{}/{}/{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 64, # NOTE: 64 for UNCONDITIONAL and 32 for CONDITIONAL
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 50,
            "threads": -1,
            "interactive": False,
            "use_ctx": False,
            "synthetic": False
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # NOTE: This is for easy recording of metrics across train, dev, test sets for each relation
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
        MRR, Precision, Precision1 = run_evaluation(args, relation['relation'], shuffle_data=False, model=model, use_ctx=args.use_ctx, synthetic=args.synthetic)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        # NOTE: This is for easy recording of metrics across train, dev, test sets for each relation
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

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_post, data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    # data_path_pre += "TREx/"
    data_path_pre = "data/LMAT/TREx_R_bench_uncond"
    # data_path_post = "train.jsonl"
    # data_path_post = "val.jsonl"
    # data_path_post = "test.jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        # {"relation": "place_of_birth", "template": "[X] was born in [Y] ."},
        # {"relation": "place_of_birth", "template": "[X] Architecture headquartered in [Y] ."},
        # {"relation": "date_of_birth", "template": "[X] (born [Y])."},
        # {"relation": "date_of_birth", "template": "Waters Societytara [X] sic citation January [Y] Leningrad postwar matter"},
        # {"relation": "place_of_death", "template": "[X] died in [Y] ."},
        {"relation": "place_of_death", "template": "[X] Hospital in [Y] ."},
    ]
    # data_path_pre = "data/Google_RE/"
    # data_path_post = "_test.jsonl"
    data_path_pre = "data/LMAT/Google_RE"
    # data_path_post = "test.jsonl"
    data_path_post = "train.jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters, metrics, dataset_type):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, metrics, dataset_type, input_param=ip)


def print_all_relation_metrics(metrics):
    for relation in metrics:
        rel_train = metrics[relation]['train']
        rel_dev = metrics[relation]['dev']
        rel_test = metrics[relation]['test']
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


if __name__ == "__main__":

    # print("1. Google-RE")    
    # parameters = get_GoogleRE_parameters()
    # run_all_LMs(parameters)

    # print("2. T-REx")
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
