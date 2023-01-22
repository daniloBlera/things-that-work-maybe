# Named Entity Recognition

Implementing a model for the named entity recognition task
with a LSTM recurrent network. This implementation was based
on the code used for the third week of Coursera's ["Natural
Language Processing with Sequence Models" course][course].

## Usage

Install the required dependencies (preferably in a
[venv][venv] or similar fashion):

```zsh
pip install more-itertools nltk tqdm trax
```

Unzip the dataset

```zsh
unzip data.zip
```

then, run the script (be warned that it takes a few seconds
for the required modules to load, before you start thinking
that the script is frozen)

```zsh
python classify.py
```

## Extra info

This script was written under a Python 3.10.9 environment,
instealled with `pyenv`. The exact version of every
dependency can be found in the `requirements.txt` file

[course]: https://www.coursera.org/learn/sequence-models-in-nlp?specialization=natural-language-processing
[venv]: https://docs.python.org/3/library/venv.html
