#!/usr/bin/env python3
"""To expose the necessary"""
print('* Importing modules, this will take a couple of seconds...')
from tagger.tools import (
        get_item_idx_maps,
        get_dataset_tensors,
        create_model,
        train_model,
        test_model_accuracy,
        predict_tags
)
