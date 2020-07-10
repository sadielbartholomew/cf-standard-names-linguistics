from pprint import pprint
from collections import Counter

from textblob import TextBlob
from textblob import formats

import matplotlib.pyplot as plt


cutoff_total = 15
ngram_n_max_interesting = 15  # at this length, mostly get count of 1  ###


class NewLineDelimitedFormat(formats.DelimitedFormat):
    ''' Allow names separated by newlines as sentence boundary markers.

    By default the textblob library uses dots as means for sentence boundary
    disambiguation, given its NLP motivation.
    '''
    delimiter = '\n'


def get_ngram_counts(names, ngram_size, get_x_most_common=None):
    '''TODO
    '''

    # Note: can't simply do:
    # ngram = names.ngrams(ngram_size)
    # as it takes ngrams across sentences, i.e. start & end of names in order.
    names_sentences = names.sentences
    ngrams = []
    for name in names_sentences:
        ngrams.extend(name.ngrams(ngram_size))

    ngram_counts = Counter(
        [tuple(wordlist) for wordlist in ngrams])  # need tuple for hashability

    if get_x_most_common:
        ngram_counts = dict(ngram_counts.most_common(get_x_most_common))
    else:
        dict(ngram_counts)

    # Format to get phrase instead of tuple of words as keys:
    ngram_counts = {
        " ".join(phrase): count for phrase, count in ngram_counts.items()}
    return ngram_counts


def plot_nrgam_counts(ngram_counts, cutoff_number, ngram_n):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    names = list(ngram_counts.keys())
    counts = list(ngram_counts.values())
    ax.barh(names, counts)

    # The n-gram names can be rather long, and often get cut off from the left
    # side of the bar plot, so instead plot them inside the bars themselves:
    # much clearer to read and compare:
    for i, (name, count) in enumerate(ngram_counts.items()):
        label = "{} ({})".format(name, count)
        plt.text(
            s=label, x=count, y=i,
            color="black", verticalalignment="center",
            horizontalalignment="right", size=10,
        )
    plt.yticks([])

    # Labelling:
    ax.set_title(
        '{} most common n-grams of size {} for the CF Standard Names'.format(
            cutoff_number, ngram_n)
    )
    ax.set_xlabel('Frequency across all standard names in current table')

    fig.set_size_inches(12, 6)
    plt.savefig(
        "../results/ngrams/most-common-{}-grams.png".format(ngram_n), dpi=400)
    plt.show()


# Grab all of the newline-delimited names (with underscores replaced by spaces)
with open("../data/all-raw-names-10.07.20-gen.txt", "r") as all_raw_names:
    names = all_raw_names.readlines()


# Replace newline with dot for standard NLP sentence delimiter, to simplify.
# TODO: use NewLineDelimitedFormat rather than manual join... not obvious how?
# formats.register('nd', NewLineDelimitedFormat)
names = ". ".join(names)
names = names.replace("\n", "")

# Create the textblob core object, a TextBlob, of all the names:
names_object = TextBlob(names)

# Find, plot and save the most common ngrams from 2 (bigram) to cutoff number:
for ngram_size in range(2, ngram_n_max_interesting + 1):
    # Get counts and conduct an assertion check
    counts = get_ngram_counts(names_object, ngram_size, cutoff_total)
    try:
        counts['azimuth angstrom']  # implies ngrams taken across names (bad)
        raise AssertionError("ngrams taken across name boundaries")
    except KeyError:
        pass  # no such key (good/correct)

    plot_nrgam_counts(counts, cutoff_total, ngram_size)
