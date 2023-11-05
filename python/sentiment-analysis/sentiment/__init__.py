# -*- coding: utf-8 -*-
"""The tools needed to train and test a sentiment classifier"""

print(f'* Loading modules, this might take a couple of seconds...')

# Exposing only what's necessary for a running example
from sentiment.tools import (
    get_positive_negative_tweets,
    tokenize_tweets,
    get_vocab,
    get_token_index_maps,
    create_classifier,
    get_train_task,
    get_eval_task,
    train_model,
    test_batch_generator,
    test_model,
    predict_sentiment
)
