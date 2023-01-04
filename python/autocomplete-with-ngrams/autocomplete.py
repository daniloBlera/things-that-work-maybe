#!/usr/bin/env python3
"""The main entry point"""
from argparse import ArgumentParser
from collections import Counter
import os

import nltk
from sklearn.model_selection import train_test_split
import tqdm

import ngrams as ng


# Type hints
sentence = list[str]
sentences = list[sentence]


def get_data_from(fpath: str,
                  test_size: float = 0.2
                  ) -> tuple[sentences, sentences, Counter]:
    """Create the training/test dataset and print a summary of them."""
    all_sentences = ng.get_tokenized_sentences_from(fpath)
    train_sentences, test_sentences = train_test_split(all_sentences,
                                                       test_size=test_size)

    print("* {} data are split into {} train and {} test set".format(
        len(all_sentences), len(train_sentences), len(test_sentences)))

    print("** First training sample:")
    print(train_sentences[0], end='\n\n')

    print("** First test sample")
    print(test_sentences[0], end='\n\n')

    oov_train_sents, oov_test_sents, vocabulary = ng.preprocess_data(
        train_sentences, test_sentences, 2)

    print("** First preprocessed training sample:")
    print(oov_train_sents[0], end='\n\n')

    print("** First preprocessed test sample:")
    print(oov_test_sents[0], end='\n\n')

    print("** 20 most common tokens from the vocabulary:")
    print('         token count')
    for (idx, (token, count)) in enumerate(vocabulary.most_common(20)):
        print(f'{idx+1:>2}: {token:>10} {count:>5}')
    print()

    print("** Size of the vocabulary:", len(vocabulary))

    return (oov_train_sents, oov_test_sents, vocabulary)


def print_suggestions(previous_tokens: list[str],
                      ngram_counts_list: list[Counter],
                      vocab_words: list[str],
                      start_with: str | None = None,
                      k: float = 1.0):
    """Print autocomplete suggestions alog with their probabilities."""
    suggestions = ng.get_suggestions(previous_tokens, ngram_counts_list,
                                     vocab_words, k=k, start_with=start_with)

    print(f'* The previous words are {previous_tokens}, the suggestions are:')
    print('   <token>:     <prob>')
    for (token, prob) in suggestions:
        print(f'{token:>10}: {prob:.8f}')

    print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('FILE', type=str, help='The path to a text document')
    parser.add_argument('--resources', type=str, default='./nltk-resources',
                        help='The path to nltk resources')
    args = parser.parse_args()

    nltk.data.path.insert(0, args.resources)
    if not os.path.exists(args.resources):
        nltk.download('punkt', download_dir=args.resources)

    train_sents, test_sents, vocabulary = get_data_from(args.FILE, 0.2)

    # Testing n-grams
    print('* Creating the list of n-grams counts...')
    ngram_counts_list = [ng.count_ngrams(train_sents, n)
                         for n in tqdm.trange(1, 6, ascii=True)]

    vocab_words = list(vocabulary.keys())
    print('Done', end='\n\n')

    # TEST 1
    previous_tokens = ["i", "am", "to"]
    print_suggestions(previous_tokens, ngram_counts_list, vocab_words)

    # TEST 2
    previous_tokens = ["i", "want", "to", "go"]
    print_suggestions(previous_tokens, ngram_counts_list, vocab_words)

    # TEST 3
    previous_tokens = ["hey", "how", "are"]
    print_suggestions(previous_tokens, ngram_counts_list, vocab_words)

    # TEST 4
    previous_tokens = ["hey", "how", "are", "you"]
    print_suggestions(previous_tokens, ngram_counts_list, vocab_words)

    # TEST 6
    previous_tokens = ["hey", "how", "are", "you"]
    print_suggestions(previous_tokens, ngram_counts_list, vocab_words,
                      start_with='d')
