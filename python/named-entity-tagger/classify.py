#!/usr/bin/env python3
"""An example of training a model for Named Entity Recogintion

This was heavily based on the implementation used for the third week of
Coursera's "Natural Language Processing with Sequence Models" course.
"""
import tagger


# Building the tokens and NER tags mapping relations to indexes
#
# The vocabulary file includes the special 'UNK' and '<pad>' strings for
# unknown and padding tokens, respectively.
(token2idx, idx2token) = tagger.get_item_idx_maps('./large/words.txt')
(tag2idx, idx2tag) = tagger.get_item_idx_maps('./large/tags.txt')

# Reading the training, validation and test (integer-encoded) datasets
#
# The number of samples per dataset
#   * 33_570 training samples
#   *  7_194 validation samples
#   *  7_194 test samples
(train_input_tensors, train_label_tensors) = tagger.get_dataset_tensors(
    './large/train/sentences.txt', './large/train/labels.txt',
    token2idx, tag2idx)

(val_input_tensors, val_label_tensors) = tagger.get_dataset_tensors(
    './large/val/sentences.txt', './large/val/labels.txt',
    token2idx, tag2idx)

(test_input_tensors, test_label_tensors) = tagger.get_dataset_tensors(
    './large/test/sentences.txt', './large/test/labels.txt',
    token2idx, tag2idx)

vocab_size = len(token2idx)
embedding_dim = 50
num_tags = len(tag2idx)
model = tagger.create_model(vocab_size, embedding_dim, num_tags)

# Training the model
print('* Training the model...')
max_steps = 1000
padding_index = token2idx['<pad>']
tagger.train_model(model, train_input_tensors, train_label_tensors,
                   val_input_tensors, val_label_tensors,
                   max_steps, padding_index)

# Testing the model with the test dataset
test_accuracy = tagger.test_model_accuracy(
    test_input_tensors, test_label_tensors, padding_index, model)
print(f'* Accuracy on the test dataset: {test_accuracy}')

# Predicting entity tags for a sentence
print('* Testing the tagger with a sentence')
sentence = '''Peter Navarro, the White House director of trade and
manufacturing policy of U.S, said in an interview on Sunday morning
that the White House was working to prepare for the possibility of a
second wave of the coronavirus in the fall, though he said it wouldn’t
necessarily come'''

for (token, tag) in tagger.predict_tags(sentence, model, token2idx, idx2tag):
    print(f'{token:>15}: {tag}')
