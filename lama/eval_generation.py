# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics
import os, json, random
import numpy as np

MAX_CONTEXT_LEN = 100


def main(args):

    if not args.text and not args.interactive:
        msg = "ERROR: either you start LAMA eval_generation with the " \
              "interactive option (--i) or you pass in input a piece of text (--t)"
        raise ValueError(msg)

    stopping_condition = True

    print("Language Models: {}".format(args.models_names))

    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)

    vocab_subset = None
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print("common vocabulary size: {}".format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    while stopping_condition:
        if args.text:
            text = args.text
            stopping_condition = False
        else:
            text = input("insert text:")

        if args.split_sentence:
            import spacy
            # use spacy to tokenize input sentence
            nlp = spacy.load(args.spacy_model)
            tokens = nlp(text)
            print(tokens)
            sentences = []
            for s in tokens.sents:
                print(" - {}".format(s))
                sentences.append(s.text)
        else:
            sentences = [text]

        if len(sentences) > 2:
            print("WARNING: only the first two sentences in the text will be considered!")
            sentences = sentences[:2]

        for model_name, model in models.items():
            print("\n{}:".format(model_name))
            original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=False)

            index_list = None
            if vocab_subset is not None:
                # filter log_probs
                filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
                filtered_log_probs_list = model.filter_logprobs(original_log_probs_list, filter_logprob_indices)
            else:
                filtered_log_probs_list = original_log_probs_list

            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                evaluation_metrics.get_ranking(filtered_log_probs_list[0], masked_indices, model.vocab, index_list=index_list)

            # prediction and perplexity for the whole softmax
            print_sentence_predictions(original_log_probs_list[0], token_ids, model.vocab, masked_indices=masked_indices)


def main_too(args):

    rel_file = 'data/relations.jsonl'
    test_data_dir = 'data/LMAT/TREx_test'
    out_dir = 'out/eval_gen/cond/rand_X5Y_cand10_bench'
    num_samples_per_rel = 10
    is_cond = True # Use context sentence
    sent_word_counts = []
    num_correct = 0
    num_samples = 0

    print("Language Models: {}".format(args.models_names))

    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)

    vocab_subset = None
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print("common vocabulary size: {}".format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    # Gather each relation's prompt
    rel_map = {}
    with open(rel_file, 'r') as f_in:
        lines = f_in.readlines()
        for line in lines:
            line = json.loads(line)
            rel = line['relation']
            prompt = line['template']
            rel_map[rel] = prompt

    for f in os.listdir(test_data_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.jsonl'):
            rel_name = os.path.basename(filename).replace('.jsonl', '')
            print('Evaluating generation for {}'.format(rel_name))
            filepath = os.path.join(test_data_dir, filename)
            with open(filepath, 'r') as f_in:
                lines = f_in.readlines()
                # Shuffle samples
                random.shuffle(lines)
                count = 0
                # for i in range(10):
                for line in lines:
                    rand_sample = json.loads(line)
                    sub_label = rand_sample['sub_label']
                    obj_label = rand_sample['obj_label']
                    context = ''

                    if is_cond:
                        evidences = rand_sample['evidences']
                        # Randomly pick a context sentence
                        rand_pair = random.choice([(evidence['obj_surface'], evidence['masked_sentence']) for evidence in evidences])
                        obj_surface, masked_sent = rand_pair
                        words = masked_sent.split()

                        # Keep track of word counts per sentence so we can calculate the average at the end
                        sent_word_counts.append(len(words))

                        if len(words) > MAX_CONTEXT_LEN:
                            # If context is too long, use the first X tokens (it's ok if obj isn't included)
                            masked_sent = ' '.join(words[:MAX_CONTEXT_LEN])
                        # If truncated context sentence still has MASK, we need to replace it with object surface but if it left out MASK, it's fine
                        context = masked_sent.replace('[MASK]', obj_surface)

                    # Build the prompt to feed into the model
                    prompt = rel_map[rel_name].replace('[X]', sub_label).replace('[Y]', '[MASK]')
                    if is_cond:
                        prompt = context + ' ' + prompt
                    sentences = [prompt]

                    if len(sentences) > 2:
                        print("WARNING: only the first two sentences in the text will be considered!")
                        sentences = sentences[:2]

                    for model_name, model in models.items():
                        print("\n{}:".format(model_name))
                        original_log_probs_list, [token_ids], [masked_indices] = model.get_batch_generation([sentences], try_cuda=False)

                        index_list = None
                        if vocab_subset is not None:
                            # filter log_probs
                            filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
                            filtered_log_probs_list = model.filter_logprobs(original_log_probs_list, filter_logprob_indices)
                        else:
                            filtered_log_probs_list = original_log_probs_list

                        # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
                        if masked_indices and len(masked_indices) > 0:
                            evaluation_metrics.get_ranking(filtered_log_probs_list[0], masked_indices, model.vocab, index_list=index_list)

                        msg_pre = "\n"
                        msg_pre += 'Subject: {}\n'.format(sub_label)
                        msg_pre += 'Gold Object: {}\n'.format(obj_label)
                        msg_pre += 'Prompt: {}\n'.format(prompt)

                        # prediction and perplexity for the whole softmax
                        perp, msg_post, pred_obj = print_sentence_predictions(original_log_probs_list[0], token_ids, model.vocab, masked_indices=masked_indices)

                        msg_pre += 'Pred Object: {}\n'.format(pred_obj)

                        # Keep track of accuracy
                        if obj_label == pred_obj:
                            num_correct += 1
                        num_samples += 1

                        # Skip samples where predicted object matches gold object
                        # if obj_label != pred_obj:
                        # Make directories in path if they don't exist
                        filepath = os.path.join(out_dir, rel_name + '.txt')
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        with open(filepath, 'a+') as f_out:
                            f_out.write(msg_pre + msg_post + '\n')
                        count += 1

                    # Move on to next relation once we hit 10 samples for a relation
                    if count >= num_samples_per_rel:
                        break

    print('Average word count of context sentences:', np.mean(sent_word_counts))
    print('Accuracy: {} / {}'.format(num_correct, num_samples))

if __name__ == '__main__':
    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    main(args)
    # main_too(args) # doesn't need text as argument
