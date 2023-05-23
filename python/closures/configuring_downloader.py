#!/usr/bin/env python3
"""An example of closure usage"""
from typing import Callable
import nltk


def make_downloader(download_path: str) -> Callable[[str], None]:
    """Return a download function configured with a download path"""
    def dl_func(package: str) -> None:
        nltk.download(package, download_dir=download_path)

    print(f'* Configuring download function to "{download_path}"')
    if download_path not in nltk.data.path:
        print(f'** Inserting "{download_path}" to nltk\'s path')
        nltk.data.path.insert(0, download_path)

    return dl_func


# configure two downloaders for different directory because REASONS!
downloader1 = make_downloader('./nltk-resources1')
downloader2 = make_downloader('./nltk-resources2')

# downloading packages into different directories
downloader1('punkt')
downloader2('twitter_samples')
downloader1('stopwords')

# using nltk.word_tokenize to check if the "punkt" package is found
sentence = 'This is an example of a text sentence.'
tokens = nltk.word_tokenize(sentence)
print(f'* sentence: {repr(sentence)}')
print(f'* tokens:   {tokens}')
