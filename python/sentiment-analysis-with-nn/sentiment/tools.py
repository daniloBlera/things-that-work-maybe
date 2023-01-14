# -*- coding: utf-8 -*-
"""An example of sentiment analysis with a two-layer neural network"""
# TODO: refactor functions into more consistent parameter order (e.g.:
# positive data, negative data, batch size, token-idx map, cycle, shuffle)
import re
import random
import string
import typing

import more_itertools as mit
import nltk
import trax
import trax.layers as tl
import jax.numpy as np
import jax
import tqdm


NLTK_DIR = 'nltk-resources'
nltk.data.path.insert(0, NLTK_DIR)


def download_package(package: str, download_dir: str = NLTK_DIR) -> None:
    """Download a package and insert the download dir into nltk's path"""
    nltk.download(package, download_dir=download_dir)


download_package('twitter_samples')
download_package('punkt')
download_package('stopwords')


TOKENIZER = nltk.tokenize.TweetTokenizer(preserve_case=False,
                                         strip_handles=True,
                                         reduce_len=True)
STEMMER = nltk.stem.PorterStemmer()
STOPWORDS = nltk.corpus.stopwords.words('english')

# Special token markers
UNKNOWN_TOKEN = '__UNK__'
PADDING_TOKEN = '__PAD__'
END_OF_TEXT_TOKEN = '__</e>__'

# For typing
T = typing.TypeVar('T')
U = typing.TypeVar('U')

# FOR DIAGNOSTICS #############################################################
# To desambiguate a default argument from a user-provided one
__DEFAULT_ARG = object()


def find_if(predicate, iterable, default=__DEFAULT_ARG, /):
    """Return the first element where `predicate(item)` is True

    Usage examples:
        # find the first even number
        find_if(lambda x: x % 2 == 0, [1, 2, 3, 4, 5]) => 2

        # find the first even number (raises ValueError)
        find_if(lambda x: x % 2 == 0, [1, 3, 5]) # raises ValueError

        # find the first even number, otherwise return None
        find_if(lambda x: x % 2 == 0, [1, 3, 5], None) => None

    Arguments:
        predicate:
            A function of one argument that returns a generalized boolean.

        iterable:
            An iterable of things.

        default:
            If unset, this function will raise a ValueError if no element
            from `iterable` passes the `predicate`. Otherwise, return the
            value given without raising ValueError.

    Return
        The first element from `iterable` that passes the `predicate`, if
        it exists.
    """
    if default is __DEFAULT_ARG:
        return mit.first(filter(predicate, iterable))
    else:
        return mit.first(filter(predicate, iterable), default)
# FOR DIAGNOSTICS #############################################################


def get_positive_negative_tweets() -> tuple[list[str], list[str]]:
    """Return positive and negative tweets

    Return a pair with the list of positive tweet texts in the first position
    and the negative tweet texts in the second.

    Return: tuple[list[str], list[str]]
        Two lists, positive and negative tweet texts (in this order), in a
        tuple.
    """
    pos_tweets = nltk.corpus.twitter_samples.strings('positive_tweets.json')
    neg_tweets = nltk.corpus.twitter_samples.strings('negative_tweets.json')
    return (pos_tweets, neg_tweets)


def tokenize_tweet(tweet: str) -> list[str]:
    """Clean and tokenize a tweet text"""
    # remove stock market tickers like $GE
    clean_tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    clean_tweet = re.sub(r'^RT[\s]+', '', clean_tweet)

    # remove hyperlinks
    clean_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', clean_tweet)

    # remove hash signs
    clean_tweet = re.sub(r'#', '', clean_tweet)
    tokens = TOKENIZER.tokenize(clean_tweet)
    tokens_clean = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS
                    and t not in string.punctuation]

    return tokens_clean


def tokenize_tweets(tweets: list[str]) -> list[list[str]]:
    """Clean and tokenize all tweet texts"""
    return [tokenize_tweet(tweet) for tweet in tqdm.tqdm(tweets, ascii=True)]


def get_vocab(tokenized_samples: list[list[str]]) -> set[str]:
    """Get all unique tokens from the tokenized text samples"""
    return {token for sent in tokenized_samples for token in sent}


def get_token_index_maps(vocab: set[str],
                         unknown_token: str = UNKNOWN_TOKEN,
                         padding_token: str = PADDING_TOKEN,
                         end_of_text_token: str = END_OF_TEXT_TOKEN
                         ) -> tuple[dict[str, int],
                                    dict[int, str]]:
    """Get the token and index mapping dicts"""
    tokens = [unknown_token, padding_token, end_of_text_token] + sorted(vocab)
    idx2token = {idx: token for (idx, token) in enumerate(tokens)}
    token2idx = {token: idx for (idx, token) in idx2token.items()}

    return (token2idx, idx2token)


def tweet_to_tensor(tweet: str,
                    token2idx: dict[str, int]
                    ) -> list[int]:
    """Return the list of token indexes from a tweet text

    Tokenize a tweet text and return the indexes mapped to each token,
    according to the token to index mapping dictionary.

    Arguments:
        tweet: str
            A tweet in text form.

        token2idx: dict[str, int]
            A token to index mapping.

    Return: list[int]
        The list of indexes related to the tokens of the tweet text.
    """
    unk_idx = token2idx[UNKNOWN_TOKEN]
    tensor = [token2idx.get(token, unk_idx) for token in tokenize_tweet(tweet)]
    return tensor


def tensor_to_tokens(tensor: list[int] | jax.Array,
                     idx2token: dict[int, str]
                     ) -> list[str]:
    """Return a list of tokens relative to the list of indexes.

    Arguments:
        tensor: list[int] | jax.Array
            A list/array of token indexes.

        idx2token: dict[int, str]
            An index to token mapping.

    Returns: list[str]
        The list of tokens relative to the list of indexes.
    """
    if isinstance(tensor, jax.Array):
        return [idx2token[idx.item()] for idx in tensor]
    else:
        return [idx2token[idx] for idx in tensor]


def pad_right(iterable: list[T], length: int, pad_element: T) -> list[T]:
    """Pad iterable if its shorter than `length`"""
    iterlen = len(iterable)

    if iterlen < length:
        return iterable + [pad_element] * (length - iterlen)
    else:
        return iterable


def batch_generator(data_pos: list[str],
                    data_neg: list[str],
                    batch_size: int,
                    token2idx: dict[str, int],
                    cycle: bool = True,
                    shuffle: bool = True,
                    padding_token: str = PADDING_TOKEN
                    ) -> typing.Generator[
                        tuple[jax.Array,
                              jax.Array,
                              jax.Array],
                        None,
                        None]:
    """The generic batch generator function used for training and testing.

    Yield (index-encoded) inputs, labels and sample weights batches with equal
    ammounts of positive and negative samples.

    Arguments:
        data_pos: list[str]
            The list of all positive samples.

        data_neg: list[str]
            The list of all negative samples.

        batch_size: int
            The size each batch should have. Incomplete batches are ignored.

        token2idx: dict[str, int]
            A token to index mapping.

        cycle: bool, optional
            If the dataset should be used in cycles. The default is True.

        shuffle: bool, optional
            If the dataset should be shuffled at the beginning of every
            epoch/cycle. The default is False.

    Return: tuple[jax.Array,
                  jax.Array,
                  jax.Array]
        Batches of (index-encoded) inputs, labels and sample weights in a
        three-items tuple (also known as a "triple"). Each of the arrays have
        the following shapes:

                    inputs.shape => (batch_size, maxlen)
                   targets.shape => (batch_size,)
            sample_weights.shape => (batch_size,)

        where `maxlen` is the length of the longest tokenized tweet in the
        batch, which means, the batches will have variable length tweets.
    """
    assert len(data_pos) == len(data_neg), f'''Error: the size of the \
positive and negative data differ \
({len(data_pos)=} != {len(data_neg)=})'''
    assert 2 <= batch_size <= len(data_pos), f'''Error: the batch size must \
be a number in the range [2, len(pos_data | neg_data)] ({batch_size=}, \
{len(data_pos)=})'''
    assert batch_size % 2 == 0, f'''Error: the batch size must be an even \
number ({batch_size=})'''

    pad_idx = token2idx[padding_token]
    half_size = batch_size // 2
    if shuffle:
        data_pos = data_pos.copy()
        data_neg = data_neg.copy()

    while True:
        if shuffle:
            random.shuffle(data_pos)
            random.shuffle(data_neg)

        pos_batches = mit.grouper(data_pos, half_size, incomplete='ignore')
        neg_batches = mit.grouper(data_neg, half_size, incomplete='ignore')
        for pos_batch, neg_batch in zip(pos_batches, neg_batches):
            tensors = [tweet_to_tensor(t, token2idx)
                       for t in pos_batch + neg_batch]

            # Skip batches with empty strings after tweet cleaning
            if [] in tensors:
                continue

            maxlen = max([len(tensor) for tensor in tensors])
            padded_tensors = [pad_right(t, maxlen, pad_idx) for t in tensors]

            inputs = np.array(padded_tensors, dtype=np.int32)
            targets = np.concatenate(
                (np.ones(half_size), np.zeros(half_size))).astype(np.int32)
            sample_weights = np.ones(batch_size, dtype=np.int32)
            yield (inputs, targets, sample_weights)

        if not cycle:
            break


def train_batch_generator(data_pos: list[str],
                          data_neg: list[str],
                          batch_size: int,
                          token2idx: dict[str, int],
                          cycle: bool = True,
                          shuffle: bool = True) -> typing.Generator:
    """The training batch generator function for the training procedure"""
    return batch_generator(
        data_pos, data_neg, batch_size, token2idx, cycle, shuffle)


def val_batch_generator(data_pos: list[str],
                        data_neg: list[str],
                        batch_size: int,
                        token2idx: dict[str, int],
                        cycle: bool = True,
                        shuffle: bool = True) -> typing.Generator:
    """The validation batch generator function for the training procedure"""
    return batch_generator(
        data_pos, data_neg, batch_size, token2idx, cycle, shuffle)


def test_batch_generator(data_pos: list[str],
                         data_neg: list[str],
                         batch_size: int,
                         token2idx: dict[str, int],
                         cycle: bool = False,
                         shuffle: bool = True) -> typing.Generator:
    """The testing batch generator function for the training procedure"""
    return batch_generator(
        data_pos, data_neg, batch_size, token2idx, cycle, shuffle)


def create_classifier(vocab_size: int = 9088,
                      embedding_dim: int = 256,
                      output_dim: int = 2) -> tl.Serial:
    """Return a sentiment classifier model

    Arguments:
        vocab_size: int, optional
            The size of the vocabulary used, including the special tokens. The
            default is 9088.
        embedding_dim: int, optional
            The size of the word embedding dimension. The default is 256.
        output_dim: int, optional
            The number of output items. The default is 2.

    Return: trax.layers.Serial
        The classification model as a series of layers.
    """
    model = tl.Serial(
        tl.Embedding(vocab_size=vocab_size, d_feature=embedding_dim),
        tl.Mean(axis=1),
        tl.Dense(n_units=output_dim),
        tl.LogSoftmax()
    )

    return model


def get_train_task(positive_samples: list[str],
                   negative_samples: list[str],
                   token2idx: dict[str, int],
                   cycle: bool = True,
                   batch_size: int = 16
                   ) -> trax.supervised.training.TrainTask:
    """Create a training task

    Arguments:
        positive_samples: list[str]
            The list of all (training) positive samples.

        negative_samples: list[str]
            The list of all (training) positive samples.

        token2idx: dict[str, int]
            A token to index mapping.

        cycle: bool, optional
            If the dataset should be used in cycles. The default is True.

        batch_size: int
            The size each batch should have. Incomplete batches are ignored.

    Returns: trax.supervised.training.TrainTask
        A trax training task.
    """
    task = trax.supervised.training.TrainTask(
        labeled_data=train_batch_generator(positive_samples, negative_samples,
                                           batch_size, token2idx, cycle,
                                           shuffle=True),
        loss_layer=trax.layers.WeightedCategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=10
    )

    return task


def get_eval_task(positive_samples: list[str],
                  negative_samples: list[str],
                  token2idx: dict[str, int],
                  cycle: bool = True,
                  batch_size: int = 16):
    """Create a training validation task

    Arguments:
        positive_samples: list[str]
            The list of all (validation) positive samples.

        negative_samples: list[str]
            The list of all (validation) positive samples.

        token2idx: dict[str, int]
            A token to index mapping.

        cycle: bool, optional
            If the dataset should be used in cycles. The default is True.

        batch_size: int
            The size each batch should have. Incomplete batches are ignored.

    Returns: trax.supervised.training.EvalTask
        A trax training validation task.
    """
    task = trax.supervised.training.EvalTask(
        labeled_data=val_batch_generator(positive_samples, negative_samples,
                                         batch_size, token2idx, cycle=cycle,
                                         shuffle=True),
        metrics=[trax.layers.WeightedCategoryCrossEntropy(),
                 trax.layers.WeightedCategoryAccuracy()]
    )

    return task


def get_test_task(positive_samples: list[str],
                  negative_samples: list[str],
                  token2idx: dict[str, int],
                  batch_size: int = 16):
    """Create a test evaluation task

    Arguments:
        positive_samples: list[str]
            The list of all (testing) positive samples.

        negative_samples: list[str]
            The list of all (testing) positive samples.

        token2idx: dict[str, int]
            A token to index mapping.

        batch_size: int
            The size each batch should have. Incomplete batches are ignored.

    Returns: trax.supervised.training.EvalTask
        A trax test evaluation task for a trained model.
    """
    task = trax.supervised.training.EvalTask(
        labeled_data=test_batch_generator(positive_samples, negative_samples,
                                          batch_size, token2idx, cycle=False,
                                          shuffle=True),
        metrics=[trax.layers.WeightedCategoryCrossEntropy(),
                 trax.layers.WeightedCategoryAccuracy()]
    )

    return task


def train_model(classifier: trax.layers.Serial,
                train_task: trax.supervised.training.TrainTask,
                eval_task: trax.supervised.training.EvalTask,
                n_steps: int,
                output_dir: str = './model'
                ) -> trax.supervised.training.Loop:
    """Run and return a training loop

    Arguments
        classifier: trax.layers.Serial
            The model to be trained.

        train_task: trax.supervised.training.TrainTask
            The training procedure.

        eval_task: trax.supervised.training.EvalTask
            The training validation procedure.

        n_steps: int
            Stop training after `n_steps`.

        output_dir: str, optional
            Where to save the training checkpoints. The default is './model'.

    Returns: trax.supervised.training.Loop
        The training loop object itself for inspection.
    """
    training_loop = trax.supervised.training.Loop(
        classifier, train_task, eval_tasks=eval_task, output_dir=output_dir)

    training_loop.run(n_steps=n_steps)
    return training_loop


# TODO: give more descriptive symbol names
def compute_accuracy(predictions: jax.Array,
                     labels: jax.Array,
                     sample_weights: jax.Array
                     ) -> tuple[float, int, int]:
    """Compute the accuracy of predictions

    Parameters
        predictions: jax.Array
            The predictions array with shape (batch_size, output_dim).

        labels: jax.Array
            The true labels array with shape (batch_size,).

        sample_weights: jax.Array
            The sample weights array, with shape (batch_size,).

    Return: TYPE
        A triple composed of the batch prediction's accuracy, the number of
        correctly classified samples and the total number of predictions
        (the same as the batch size).
    """
    is_positive = (predictions[:, 0] < predictions[:, 1]).astype(np.int32)
    correct = (is_positive == labels).astype(np.float32)
    weighted_correct = np.multiply(sample_weights, correct)
    batch_size = len(sample_weights)
    weighted_num_correct = np.sum(weighted_correct, dtype=np.int32).item()
    accuracy = weighted_num_correct / batch_size

    return (accuracy, weighted_num_correct, batch_size)


def test_model(batch_generator: typing.Generator,
               model: trax.layers.Serial,
               compute_accuracy: typing.Callable = compute_accuracy
               ) -> float:
    """Compute the accuracy of the trained model on the test dataset

    Arguments:
        batch_generator: typing.Generator
            A test sample generator.
        model: trax.layers.Serial
            A trained trax model.
        compute_accuracy: typing.Callable, optional
            A function to compute the model accuracy on every batch. The
            default is the `compute_accuracy` function.

    Returns: float
        The accuracy of the trained model on the test dataset.
    """
    total_num_correct = 0
    total_num_predictions = 0

    #   inputs.shape => (batch_size, max_tweet_len)
    #   labels.shape => (batch_size,)
    # sample_weights => (batch_size,)
    for (inputs, labels, sample_weights) in batch_generator:
        predictions = model(inputs)
        (_, num_correct, num_samples) = compute_accuracy(predictions,
                                                         labels,
                                                         sample_weights)
        total_num_correct += num_correct
        total_num_predictions += num_samples

    return total_num_correct / total_num_predictions


def predict_sentiment(tweet: str,
                      token2idx: dict[str, int],
                      model: trax.layers.Serial,
                      ) -> typing.Literal['positive', 'negative']:
    """Predict the sentiment on a text sentence
    Parameters
    ----------
    tweet : str
        DESCRIPTION.
    token2idx : dict[str, int]
        DESCRIPTION.
    model : trax.layers.Serial
        DESCRIPTION.
    verbose : bool, optional
        DESCRIPTION. The default is False.

    Returns: typing.Literal['positive', 'negative']
        Return the predicted sentiment of the text.
    """
    input_tensor = np.array(tweet_to_tensor(tweet, token2idx))
    input_tensor = input_tensor[None, :]
    prediction = model(input_tensor)

    return 'positive' if prediction[0, 0] < prediction[0, 1] else 'negative'
