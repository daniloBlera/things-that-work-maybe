# N-grams and autocompletion

Here's an implementation of autocompletion with word-based n-gram language
models. Here are the source material used:

* [Natural Language Processing with Probabilistic Models][c2] (week 2)
* [Speech and Language Processing][book] (chapter 3)

## Usage

Install the required dependencies (preferably in a
[venv][venv] or similar fashion):

```zsh
pip install more-itertools nltk pandas scikit-learn tqdm
```

then, run the script with

```zsh
python autocomplete.py [--resources PATH/TO/NLTK/RESOURCES] en_US.twitter.txt
```

where `--resources` can be used to indicate the path to the
root of NLTK's resource downloads, for example, to download
resources into the current directory instead of the user's
home, do:

```zsh
python autocomplete.py --resources nltk-resources en_US.twitter.txt
```

## Extra info

The script written with Python 3.11.0 and whatever up-to-date versions of the
required modules at the time.

[c2]: https://www.coursera.org/learn/probabilistic-models-in-nlp?specialization=natural-language-processing
[book]: https://web.stanford.edu/~jurafsky/slp3/
[venv]: https://docs.python.org/3/library/venv.html
