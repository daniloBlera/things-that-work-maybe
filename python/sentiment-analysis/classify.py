#!/usr/bin/env python3
"""An example of training a sentiment classification model

This was heavily based on the implementation used for the first week of
Coursera's "Natural Language Processing with Sequence Models" course.
"""
import shutil
import random
from sklearn.model_selection import train_test_split
import sentiment as st


if __name__ == '__main__':
    # Reading the tweet texts
    (pos_tweets, neg_tweets) = st.get_positive_negative_tweets()

    # Splitting the dataset into training, validation and testing datasets
    # A summary of the samples:
    #
    #   * 10_000 total samples, 5_000 positive and 5_000 negative samples;
    #   *  8_000 samples will be used for training;
    #   *  1_600 samples will be used for validation during training;
    #   *    400 samples will be used for testing the trained model;
    train_pos, val_test_pos = train_test_split(pos_tweets, test_size=0.2)
    train_neg, val_test_neg = train_test_split(neg_tweets, test_size=0.2)
    val_pos, test_pos = train_test_split(val_test_pos, test_size=0.2)
    val_neg, test_neg = train_test_split(val_test_neg, test_size=0.2)
    print('\n\n* A summary of the dataset')
    print(f'** {len(pos_tweets):>4} positive samples')
    print(f'** {len(neg_tweets):>4} negative samples')
    print(f'** {len(train_pos) + len(train_neg):>4} training saples')
    print(f'** {len(val_pos) + len(val_neg):>4} validation samples')
    print(f'** {len(test_pos) + len(test_neg):>4} test samples')

    # Building the vocabulary and token-index maps from the training corpus
    print('\n* Creating a vocabulary from the training corpus...')
    tokenized_train_tweets = st.tokenize_tweets(train_pos + train_neg)
    vocab = st.get_vocab(tokenized_train_tweets)
    (token2idx, idx2token) = st.get_token_index_maps(vocab)

    # To include the special token markers for "unknown token", "padding", and
    # "end of text"
    vocab_size = len(token2idx)

    # Creating the model and the training, validation, and test tasks
    model = st.create_classifier(vocab_size)
    train_task = st.get_train_task(train_pos, train_neg, token2idx)
    val_task = st.get_eval_task(val_pos, val_neg, token2idx)
    test_task = st.get_eval_task(test_pos, test_neg, token2idx)

    # Deleting training checkpoints before every run
    modeldir = './model-checkpoints'
    try:
        shutil.rmtree(modeldir)
    except OSError:
        pass

    # Run a training loop with the training data and report the performance
    # with the validation data
    print('\n* Training the classifier model...')
    training_loop = st.train_model(model, train_task, val_task, 100, modeldir)

    # Evaluate the damn thing with the test dataset
    test_batches = st.test_batch_generator(test_pos, test_neg, 64, token2idx)
    accuracy = st.test_model(test_batches, model)
    print(f'\n* Accuracy on the test dataset: {accuracy}')

    # Predicting with a random positive sample from the test dataset
    tweet = random.choice(test_pos)
    print(f'\n* tweet: {repr(tweet)}')
    print(f'* Prediction: {st.predict_sentiment(tweet, token2idx, model)}')

    # Predicting with a random negative sample from the test dataset
    tweet = random.choice(test_neg)
    print(f'\n* tweet: {repr(tweet)}')
    print(f'* Prediction: {st.predict_sentiment(tweet, token2idx, model)}')

    # Trying predicting some sentences
    sentence = '''It's such a nice day, I think I'll be taking Sid to \
Ramsgate for lunch and then to the beach maybe.'''
    print('\n* Sentence')
    print(repr(sentence))
    print(f'* Prediction: {st.predict_sentiment(sentence, token2idx, model)}')

    sentence = "I hated my day, it was the worst, I'm so sad."
    print('\n* Sentence')
    print(repr(sentence))
    print(f'* Prediction: {st.predict_sentiment(sentence, token2idx, model)}')
