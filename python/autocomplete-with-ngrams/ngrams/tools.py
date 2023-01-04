#!/usr/bin/env python3
"""Everything and the kitchen sink

All the tools needed to implement an N-Gram language model for autocompletion.
"""
import collections
import functools
import itertools
import typing
import re

import more_itertools
import nltk
import numpy
import pandas
import tqdm


# Type aliases
token = str
sentence = list[token]


# A sentinel for default arguments different than None. This is for internal
# usage only
__SENTINEL = object()

UNKNOWN_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '<e>'
SMOOTHING_CONST = 1.0


# Utils that shouldn't be imported
def logcall(func):
    """Print to stdout the function object and its arguments

    Decorate a function with a logger that prints the function's object
    as well as its arguments.
    """
    def wrapper(*args, **kwargs):
        print(f'{func=}')
        print(f'{args=}')
        print(f'{kwargs=}')
        return func(*args, **kwargs)

    return wrapper


def find_if(predicate, iterable, default=__SENTINEL, /):
    """Return the first item where `predicate(item)` is True

    Example: find the first tokenized sentence that has a '<unk>' token
        find_if(lambda s: '<unk>' in s, tokenized_sentences)
    """
    if default is __SENTINEL:
        return more_itertools.first(filter(predicate, iterable))
    else:
        return more_itertools.first(filter(predicate, iterable), default)


# Shorthands for common slicing to avoid off-by-one errors
def lastn(iterable, n=1):
    assert n >= 1, 'Error: n must be a positive integer'
    return iterable[-n:]


# The useful stuff that should be imported
def get_tokenized_sentences_from(filepath: str) -> list[sentence]:
    """Create a list of tokenized sentences from a text file."""
    with open(filepath) as fd:
        print(f"* Reading sentences from '{filepath}'")
        text = fd.read().lower()
        text = re.sub(r'\n\n+', '', text)
        lines = tqdm.tqdm(text.splitlines(), ascii=True)
        sents = [nltk.word_tokenize(sent) for sent in lines]
        print('* Done', end='\n\n')
        return sents


def count_tokens(tokenized_sentences: list[sentence],
                 count_threshold: int = 1) -> collections.Counter:
    """Return a counter of all the tokens in a corpus

    Arguments:
        tokenized_sentences: list[sentence]
            a list of tokenized sentences.

        count_threshold: int
            ignore words with counts less than this threshold.

    Return: collections.Counter[token, int]
        A counter instance with the tokens and their respective absolute
        frequencies.
    """
    all_tokens = [t for sent in tokenized_sentences for t in sent]
    counts = collections.Counter(all_tokens)
    return collections.Counter({w: c for (w, c) in counts.items()
                                if c >= count_threshold})


def replace_oov_tokens(tokenized_sents: list[sentence],
                       closed_vocab: collections.Counter,
                       unknown_token: str = UNKNOWN_TOKEN
                       ) -> list[sentence]:
    """Replace OOV tokens by a specific unknown token flag.

    Given a list of tokenized sentences and a vocabulary, return a copy
    of the sentences with all out-of-vocabulary (OOV) tokens replaced by
    the unknown token indicator.

    Arguments:
        tokenized_sents: list[sentence],
            A list of tokenized sentences.

        closed_vocab: collections.Counter[token, int]
            A counter with a corpus vocabulary and their respective absolute
            frequencies.

        unknown_token: str = UNKNOWN_TOKEN
            The marker to substitute tokens missing from `closed_vocab`.

    Return: list[sentence]
        A list of tokenized sentences with OOV tokens replaced by
        `unknown_token`.
    """
    def __replace(sent):
        """Helper to replace OOV tokens from a single sentence"""
        return [t if t in closed_vocab else unknown_token for t in sent]

    return [__replace(sent) for sent in tqdm.tqdm(tokenized_sents, ascii=True)]


# TODO: docstring
def preprocess_data(train_tokenized_sentences: list[str],
                    test_tokenized_sentences: list[str],
                    count_threshold: int,
                    unknown_token: str = UNKNOWN_TOKEN,
                    token_counter: typing.Callable = count_tokens,
                    oov_token_replacer: typing.Callable = replace_oov_tokens
                    ) -> tuple[list[sentence],
                               list[sentence],
                               collections.Counter]:
    """DOCSTRING"""
    vocabulary = token_counter(train_tokenized_sentences, count_threshold)
    print('* Replacing OOV tokens from training set...')
    oov_train_sentences = oov_token_replacer(
        train_tokenized_sentences, vocabulary, unknown_token=unknown_token)
    print('Done', end='\n\n')

    print('* Replacing OOV tokens from test set...')
    oov_test_sentences = oov_token_replacer(
        test_tokenized_sentences, vocabulary, unknown_token=unknown_token)
    print('Done', end='\n\n')

    return (oov_train_sentences, oov_test_sentences, vocabulary)


def count_ngrams(tokenized_sentences: list[sentence], n: int,
                 start_token: str = START_TOKEN, end_token: str = END_TOKEN
                 ) -> collections.Counter:
    """Count the n-grams from the tokenized sentences.

    Given a list of tokenized sentences, count the absolute frequencies of all
    n-grams in all sentences.

    Arguments:
        tokenized_sents: list[sentence],
            A list of tokenized sentences.

        n: int
            The size of the n-gram

        start_token: int = START_TOKEN
            The marker used to indicate the start of a sentence.

        end_token: str = END_TOKEN
            The marker used to indicate the end of a sentence.

    Return: collections.Counter[token, int]
        The absolute frequencies of sentence n-grams from the corpus,
        including the `start_token` and `end_token` markers.
    """
    ngrams: dict[str, int] = {}
    for sent in tokenized_sentences:
        sent = [start_token] * n + sent + [end_token]
        for ngram in nltk.ngrams(sent, n):
            ngrams[ngram] = ngrams.get(ngram, 0) + 1

    return collections.Counter(ngrams)


def estimate_probability(word: token,
                         prefix: list[token],
                         sequence_counts: collections.Counter,
                         prefix_counts: collections.Counter,
                         vocab_size: int,
                         k: float = SMOOTHING_CONST) -> float:
    """Calculate the joint probability of a word, given a prefix.

    This is the implementation of the probability of an n-gram, using the
    joint probability for the final word given a prefix.

    Arguments:
        word: token
            The last token of an n-gram.

        prefix: list[token]
            The tokens from an n-gram except for the last one.

        sequence_counts: collections.Counter[token, int]
            The counts of all complete n-grams (a prefix plus the last word).

        prefix_counts: collections.Counter[token, int]
            The counts of all prefixes (n-grams minus the last token).

        vocab_size: int
            The size of the corpus' vocabulary.

        k: float
            The smoothing constant.

    Return: 0 <= float <= 1
        The joint probability of `word` given its `prefix`.
    """
    sequence = tuple(prefix + [word])
    numerator = sequence_counts[sequence] + k
    denominator = prefix_counts[tuple(prefix)] + (k * vocab_size)

    return numerator / denominator


def estimate_probabilities(prefix: list[token],
                           sequence_counts: collections.Counter,
                           prefix_counts: collections.Counter,
                           vocabulary: list[token],
                           end_token: str = END_TOKEN,
                           unknown_token: str = UNKNOWN_TOKEN,
                           k: float = SMOOTHING_CONST
                           ) -> collections.Counter:
    """Estimate the joint probabilities for every words given the prefix.

    For every word in the vocabulary, calculate its joint probability, given
    the `prefix` from the argument.

    Arguments:
        prefix: list[token]
            The tokens from an n-gram except for the last one.

        sequence_counts: collections.Counter[token, int]
            The counts of all complete n-grams (a prefix plus the last word).

        prefix_counts: collections.Counter[token, int]
            The counts of all prefixes (n-grams minus the last token).

        vocabulary: list[token]
            The corpus' vocabulary.

        end_token: str = Token.Start
            The marker used to indicate the end of a sentence.

        unknown_token: str = UNKNOWN_TOKEN
            The marker to substitute tokens missing from the vocabulary.

        k: float = SMOOTHING_CONST
            The smoothing constant.

    Return: collections.Counter[token, float]
        The counts for joint probabilities of all words in the vocabulary given
        the prefix.
    """
    vocab = vocabulary + [end_token, unknown_token]
    vocab_size = len(vocab)

    # setting fixed arguments for a shorter alias
    #   calculate the joint probability of a word, given a prefix
    cal_prob = functools.partial(estimate_probability,
                                 prefix=prefix,
                                 sequence_counts=sequence_counts,
                                 prefix_counts=prefix_counts,
                                 vocab_size=vocab_size,
                                 k=k)

    probs = {w: cal_prob(w) for w in vocab}
    return collections.Counter(probs)


# TODO: docstring
def __get_count_dframe(sequence_counts: collections.Counter,
                       vocabulary: set[token],
                       end_token: str = END_TOKEN,
                       unknown_token: str = UNKNOWN_TOKEN
                       ) -> pandas.DataFrame:
    """DOCSTRING"""
    last_words = list(vocabulary) + [unknown_token, end_token]
    prefixes = {tuple(prefix) for (*prefix, word) in sequence_counts.keys()}
    nrows = len(prefixes)
    ncols = len(last_words)

    countdf = pandas.DataFrame(numpy.zeros((nrows, ncols)),
                               columns=last_words,
                               index=list(prefixes))

    for (prefix, word) in itertools.product(prefixes, last_words):
        countdf[word][prefix] = sequence_counts[prefix + (word,)]

    return countdf


# TODO: docstring
def get_probability_dframe(sequence_counts: collections.Counter,
                           vocabulary: set[token],
                           k: float) -> pandas.DataFrame:
    """DOCSTRING"""
    countdf = __get_count_dframe(sequence_counts, vocabulary)
    countdf += k
    return countdf.div(countdf.sum(axis=1), axis=0)


def __ngram_size(prefix_counts: collections.Counter) -> int:
    """Calculate the size of the n-grams from a prefix counter

    The structure of an n-gram is composed of two parts, a prefix and a word,
    the `prefix` is a list of tokens (zero in the case of unigrams) while
    `word` is a single token. This function returns the number of elements
    in the `prefix` part by counting the number of elements in any of the
    `prefix_counts` keys (which is a list of tokens representing the prefix).

    Arguments:
        prefix_counts: collections.Counter[token, int]
            The counts of all prefixes (n-grams minus the last token).

    Return: int
        The size of the prefixes from `prefix_count`. This is equivalent to
        counting the size of the sequence n-gram minus 1.
    """
    prefix = list(prefix_counts.keys())[0]
    return len(prefix)


# TODO: docstring
def calculate_perplexity(tokenized_sentence: sentence,
                         sequence_counts: collections.Counter,
                         prefix_counts: collections.Counter,
                         vocab_size: int,
                         start_token: str = START_TOKEN,
                         end_token: str = END_TOKEN,
                         k: float = SMOOTHING_CONST,
                         log_perplexity: bool = False) -> float:
    """DOCSTRING

    Arguments:
        tokenized_sentence: list[token]
            A single tokenized sentence.

        sequence_counts: collections.Counter[token, int]
            The counts of all complete n-grams (a prefix plus the last word).

        prefix_counts: collections.Counter[token, int]
            The counts of all prefixes (n-grams minus the last token).

        vocab_size: int
            The number of unique tokens in the corpus' vocabulary.

        start_token: int = START_TOKEN
            The marker used to indicate the start of a sentence.

        end_token: str = END_TOKEN
            The marker used to indicate the end of a sentence.

        k: float = SMOOTHING_CONST
            The smoothing constant.

        log_perplexity: bool = False
            Transform the product of probabilities into a sum of log
            probabilities from base 2.
    """
    # the size of the prefix, i.e., the size of the n-gram minus the last word
    n = __ngram_size(prefix_counts)
    sentence = tuple([start_token] * n + tokenized_sentence + [end_token])

    # Note from the book about N:
    #   We also need to include the end-of-sentence marker </s>
    #   (but not the beginning-of-sentence marker <s>) in the
    #   total count of word to- kens N.
    #
    # but here we're including the <s> * n tokens into the count
    N = len(sentence)

    # Setting fixed arguments for a shorter alias:
    #   Calculate the joint probability of a word, given a prefix
    calc_prob = functools.partial(estimate_probability,
                                  sequence_counts=sequence_counts,
                                  prefix_counts=prefix_counts,
                                  vocab_size=vocab_size,
                                  k=k)

    # For all words in the sentence, calculate the probability
    # the n-gram probability (prefix and the final word)
    ngrams = nltk.ngrams(sentence, n+1)
    if not log_perplexity:
        probs = [1 / calc_prob(w, ps) for (*ps, w) in ngrams]
        product = functools.reduce(
            lambda acc, x: acc * x, probs, 1.0)
        return numpy.power(product, 1 / N)
    else:   # TODO: check if this actually works
        probs = [calc_prob(w, ps) for (*ps, w) in ngrams]
        summation = functools.reduce(
            lambda acc, x: acc + numpy.log2(x), probs, 0)
        return -(1/N) * summation


# TODO: docstring
def suggest_word(tokenized_sentence: list[token],
                 sequence_counts: collections.Counter,
                 prefix_counts: collections.Counter,
                 vocabulary: list[token],
                 start_token: str = START_TOKEN,
                 end_token: str = END_TOKEN,
                 k: float = SMOOTHING_CONST,
                 start_with: str | None = None
                 ) -> tuple[token, float]:
    """DOCSTRING

    Arguments:
        tokenized_sentence: list[token]
            A single tokenized sentence. It must have at least N tokens, where
            N is the size of the N-gram.

        sequence_counts: collections.Counter[token, int]
            The counts of all complete n-grams (a prefix plus the last word).

        prefix_counts: collections.Counter[token, int]
            The counts of all prefixes (n-grams minus the last token).

        vocab_size: int
            The number of unique tokens in the corpus' vocabulary.

        start_token: int = START_TOKEN
            The marker used to indicate the start of a sentence.

        end_token: str = END_TOKEN
            The marker used to indicate the end of a sentence.

        k: float = SMOOTHING_CONST
            The smoothing constant.

        start_with: str | None = None
            How the suggested word should start.

    Return: tuple[token, float]
        A pair with the token with the highest probability to immediately
        succeed the tokenized sentence.
    """
    prefix = lastn(tokenized_sentence, __ngram_size(prefix_counts))
    probabilities = estimate_probabilities(
        prefix, sequence_counts, prefix_counts, vocabulary,
        end_token=end_token, k=k)

    suggestion_token = ''
    suggestion_prob = float('-inf')     # To guarantee at least one update
    for token, prob in probabilities.items():
        if start_with and not token.startswith(start_with):
            continue

        if prob > suggestion_prob:
            suggestion_token = token
            suggestion_prob = prob

    return (suggestion_token, suggestion_prob)


# TODO: docstring
def get_suggestions(previous_tokens: list[token],
                    ngram_counts_list: list[collections.Counter],
                    vocabulary: list[token],
                    k: float = SMOOTHING_CONST,
                    start_with: str | None = None
                    ) -> list[tuple[token, float]]:
    """DOCSTRING"""
    suggestions = []
    for (prefix_counts, sequence_counts) in itertools.pairwise(
            ngram_counts_list):
        suggestions.append(suggest_word(previous_tokens, sequence_counts,
                                        prefix_counts, vocabulary, k=k,
                                        start_with=start_with))

    return suggestions
