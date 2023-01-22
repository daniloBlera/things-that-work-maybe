# Classifying sentiments from tweets

Implementing a tweet sentiment classifier using a two-layer
feed-forward network. This implentation was heavily based on
the implementation used for the first week of Coursera's
["Natural Language Processing with Sequence Models"
course][c1].

## Usage

Install the required dependencies (preferably in a
[venv][venv] or similar fashion):

```zsh
pip install more-itertools nltk scikit-learn tqdm trax
```

then, run the script with (be warned that it takes a few
seconds for the required modules to load, before you start
thinking that the script is frozen):

```zsh
python classify.py
```

## Extra info

This script was written under a Python 3.10.9 environment,
instealled with `pyenv`. The exact version of every
dependency can be found in the `requirements.txt` file

[c1]: https://www.coursera.org/learn/sequence-models-in-nlp?specialization=natural-language-processing
[venv]: https://docs.python.org/3/library/venv.html
