#!/usr/bin/env python3
"""An example of closure usage in python"""
import nltk


def make_downloader(download_path):
    """Return a download function configured with a download path"""
    def dl_func(package):
        nltk.download(package, download_dir=download_path)
        if download_path not in nltk.data.path:
            nltk.data.path.insert(0, download_path)

    return dl_func


# assign to `download` a download function configured to
# use './nltk-resources' as the download directory
download = make_downloader('./nltk-resources')

# download the 'punkt', 'twitter_samples', and 'stopwords'
# packages with the previously configured download function
download('punkt')
download('twitter_samples')
download('stopwords')
