"""Everything needed for the NER task"""
import random
import shutil
from pathlib import Path
from typing import Callable, Generator, TypeVar

import jax
import jax.numpy as jnp
import more_itertools as mit
import nltk
import trax.data.inputs as inputs
import trax.layers as tl
import trax.optimizers as optim
from tqdm import tqdm
from trax.supervised import training

NLTK_DIR = 'nltk-resources'
nltk.data.path.insert(0, NLTK_DIR)


def download_package(package: str, download_dir: str = NLTK_DIR) -> None:
    """Download a package and insert the download dir into nltk's path"""
    nltk.download(package, download_dir=download_dir)


download_package('punkt')
download_package('stopwords')

# Special token markers
UNKNOWN_TOKEN = 'UNK'
PADDING_TOKEN = '<pad>'

# Generics stuff
T = TypeVar('T')
U = TypeVar('U')


def get_item_idx_maps(
        items_filepath: str
        ) -> tuple[dict[str, int], dict[int, str]]:
    unique_items = Path(items_filepath).read_text().strip().splitlines()
    idx2item = dict(enumerate(unique_items))
    item2idx = {token: idx for (idx, token) in idx2item.items()}
    return (item2idx, idx2item)


def create_converter(
        type_map: dict[T, U],
        default_key: T
        ) -> Callable[[list[T]], list[U]]:
    converter_map = type_map.copy()

    def convert(items: list[T]) -> list[U]:
        return [converter_map.get(e, default_key) for e in items]

    return convert


def get_dataset_tensors(
        inputs_filepath: str,
        labels_filepath: str,
        token_to_idx: dict[str, int],
        tag_to_idx: dict[str, int]
        ) -> tuple[list[list[int]], list[list[int]]]:
    unk_idx = token_to_idx[UNKNOWN_TOKEN]
    other_tag = tag_to_idx['O']

    # Defaults to "unknown" token (index)
    tokens_to_tensors = create_converter(token_to_idx, unk_idx)

    # Defaults to "other" tag (index)
    tags_to_tensors = create_converter(tag_to_idx, other_tag)

    print('* Reading input samples...')
    inputs = tqdm(Path(inputs_filepath).read_text().splitlines(), ascii=True)
    input_tensors = [tokens_to_tensors(nltk.word_tokenize(sent))
                     for sent in inputs]

    print('* Reading labels...')
    labels = tqdm(Path(labels_filepath).read_text().splitlines(), ascii=True)
    label_tensors = [tags_to_tensors(label.split())
                     for label in labels]

    return (input_tensors, label_tensors)


def batch_generator(
        inputs: list[list[int]],
        labels: list[list[int]],
        batch_size: int,
        padding_index: int,
        cycle: bool = True,
        shuffle: bool = True
        ) -> Generator:
    def rpad(tensor: list[int], length: int) -> list[int]:
        """Pad tensor if it's shorter than `length`"""
        if len(tensor) >= length:
            return tensor
        pad = [padding_index] * (length - len(tensor))
        return tensor + pad

    assert len(inputs) == len(labels), '''Error: size mismatch between \
inputs and labels'''

    assert batch_size <= len(labels), '''Error: batch size is greater than \
the total number samples'''

    dataset = list(zip(inputs, labels))
    while True:
        if shuffle:
            random.shuffle(dataset)

        for batch in mit.grouper(dataset, batch_size, incomplete='ignore'):
            (batch_inputs, batch_labels) = list(zip(*batch))
            maxlen = max(map(len, batch_inputs))
            batch_inputs_pad = [rpad(x, maxlen) for x in batch_inputs]
            batch_labels_pad = [rpad(y, maxlen) for y in batch_labels]
            batch_inputs_arr = jnp.array(batch_inputs_pad, dtype=jnp.int32)
            batch_labels_arr = jnp.array(batch_labels_pad, dtype=jnp.int32)
            yield (batch_inputs_arr, batch_labels_arr)

        if not cycle:
            return


def create_model(
        vocab_size: int,
        embedding_dim: int,
        num_tags: int
        ) -> tl.Serial:
    model = tl.Serial(
        tl.Embedding(vocab_size=vocab_size, d_feature=embedding_dim),
        tl.LSTM(n_units=embedding_dim),
        tl.Dense(n_units=num_tags),
        tl.LogSoftmax()
    )

    return model


def train_model(
        model: tl.Serial,
        train_inputs: list[list[int]],
        train_labels: list[list[int]],
        val_inputs: list[list[int]],
        val_labels: list[list[int]],
        max_steps: int,
        padding_index: int,
        batch_size: int = 64,
        output_dir: str = './model-checkpoints'
        ):
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        pass

    train_generator = inputs.add_loss_weights(
        batch_generator(inputs=train_inputs, labels=train_labels,
                        batch_size=batch_size, padding_index=padding_index,
                        cycle=True, shuffle=True),
        id_to_mask=padding_index
    )

    eval_generator = inputs.add_loss_weights(
        batch_generator(inputs=val_inputs, labels=val_labels,
                        batch_size=batch_size, padding_index=padding_index,
                        cycle=True, shuffle=True),
        id_to_mask=padding_index
    )

    train_task = training.TrainTask(
        labeled_data=train_generator,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=optim.Adam(0.01),
    )

    eval_task = training.EvalTask(
        labeled_data=eval_generator,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=10
    )

    training_loop = training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir
    )

    training_loop.run(n_steps=max_steps)
    return training_loop


def compute_accuracy(
        predictions: jax.Array,
        labels: jax.Array,
        padding_index: int
        ) -> float:
    # argmax'ing the softmax dimension
    # predictions.shape := (batch_size, max_length, num_classes)
    outputs = jnp.argmax(predictions, axis=2)
    mask = labels != padding_index
    accuracy = jnp.sum(outputs[mask] == labels[mask]) / jnp.sum(mask)
    return accuracy.item()


def test_model_accuracy(
        test_inputs: list[list[int]],
        test_labels: list[list[int]],
        padding_index: int,
        model: tl.Serial
        ) -> float:
    # Processing the entire test dataset in a single batch
    batch = batch_generator(
        inputs=test_inputs, labels=test_labels,
        batch_size=len(test_inputs), padding_index=padding_index,
        cycle=False, shuffle=False)

    (inputs, labels) = next(batch)
    predictions = model(inputs)
    accuracy = compute_accuracy(predictions, labels, padding_index)
    return accuracy


def predict_tags(
        sentence: str,
        model: tl.Serial,
        token_to_idx: dict[str, int],
        idx_to_tag: dict[int, str]
        ) -> list[tuple[str, str]]:
    tokens = nltk.word_tokenize(sentence)
    unk_idx = token_to_idx[UNKNOWN_TOKEN]
    input_tensor = [token_to_idx.get(t, unk_idx) for t in tokens]
    x = jnp.array(input_tensor, dtype=jnp.int32)
    x = x.reshape((1, -1))
    prediction_idxs = jnp.argmax(model(x), axis=-1).reshape((-1,))
    prediction_tags = [idx_to_tag[idx.item()] for idx in prediction_idxs]
    return list(zip(tokens, prediction_tags))
