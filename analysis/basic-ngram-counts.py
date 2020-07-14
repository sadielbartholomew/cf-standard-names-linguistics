from pprint import pprint
from collections import Counter

from textblob import TextBlob
from textblob import formats

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


cutoff_total = 15
ngram_n_max_interesting = 20  # at this length, mostly get count of 1


def get_ngram_counts(names, ngram_size, get_x_most_common=None):
    '''TODO
    '''
    # Note, can't simply do:
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


def get_and_remove_any_n_ngram_counts(names):
    '''TODO
    '''
    most_common_any_n_ngrams = []
    max_count = 0
    # This time remove ngrams as they are found!
    names_sentences = names.sentences
    # reversed(...) is important here.
    for size in reversed(range(2, ngram_n_max_interesting)):
        # Get n-grams at this size
        size_n_ngrams = []
        for name in names_sentences:
            ngrams_of_size = name.ngrams(size)
            size_n_ngrams.extend(ngrams_of_size)
        most_common_size_n = Counter(
            [tuple(wordlist) for wordlist in size_n_ngrams]).most_common(1)

        # TODO: ensure nothing draws, as will need to add both!
        ngram_max_count = most_common_size_n[0][1]

        if ngram_max_count >= max_count:  # take longer one
            most_common_any_n_ngrams.append(most_common_size_n)
            max_count = ngram_max_count

    # Determine most common over any size:
    final_most_common_any_n_ngrams = sorted(
        most_common_any_n_ngrams, key=lambda item: item[0][1], reverse=True)
    # TODO: manage this mess; need to catch counts of equal value post-sort
    if (final_most_common_any_n_ngrams[0][0][1] ==
            final_most_common_any_n_ngrams[1][0][1]):
        print("\nNote, duplicates:", final_most_common_any_n_ngrams[:2])
        if (final_most_common_any_n_ngrams[0][0][0][:-1] ==
                final_most_common_any_n_ngrams[1][0][0]):
            pass
        if (final_most_common_any_n_ngrams[1][0][1] ==
                final_most_common_any_n_ngrams[2][0][1]):
            print("\nFURTHER DUPLICATES, CHECK HOW FAR EQUAL MAX. GOES...",
                  final_most_common_any_n_ngrams)

    return final_most_common_any_n_ngrams[0], names  # any_n_ngram_counts


def recursive_get_and_remove_any_n_ngram_counts(names, get_x_most_common):
    '''TODO
    '''
    # Get most common n-gram of any size:
    overall_most_common_ngrams = []
    use_names = names
    while len(overall_most_common_ngrams) < get_x_most_common:
        packed_most_freq_ngram, all_names = get_and_remove_any_n_ngram_counts(
            use_names)
        most_common_n_gram = packed_most_freq_ngram
        overall_most_common_ngrams.append(most_common_n_gram)

        # Remove that most common n-gram from all names ready to go again:
        most_common_ngram_raw = " ".join(most_common_n_gram[0][0])
        new_names = []
        for name in all_names.sentences:
            if most_common_ngram_raw in name:
                name = name.replace(most_common_ngram_raw, "")
                name = name.replace("  ", " ")
            new_names.append(name)
        new_names = "\n".join([str(name) for name in new_names])
        use_names = TextBlob(new_names)

    # Format to get phrase instead of tuple of words as keys:
    ngram_counts = {
        entry[0][0]: entry[0][1] for entry in overall_most_common_ngrams}
    ngram_counts = {
        " ".join(phrase): count for phrase, count in ngram_counts.items()}
    print(ngram_counts)
    return ngram_counts


def plot_ngram_counts(ngram_counts, cutoff_number=None, ngram_n=None):
    '''TODO
    '''
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


def plot_recursive_ngram_counts(ngram_counts):
    '''TODO
    '''
    plt.rcdefaults()
    fig, ax = plt.subplots()
    names = list(ngram_counts.keys())
    counts = list(ngram_counts.values())
    ax.barh(names, counts, color='lightcoral')

    # Axes formatting:
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", direction="in", pad=-10)
    plt.setp(ax.get_yticklabels(), ha="left")
    for i, (name, count) in enumerate(ngram_counts.items()):
        label = "{} ({})".format(name, count)
        plt.text(
            s=label, x=count, y=i,
            color="black", verticalalignment="center",
            ha="left", size=10,
        )
    plt.yticks([])

    # Labelling:
    ax.set_title(
        '{} most common n-grams of *any* size for the '
        'CF Standard Names'.format(len(names))
    )
    ax.set_xlabel('Frequency across all standard names in current table')

    fig.set_size_inches(12, 12)
    plt.savefig(
        "../results/ngrams/recursive-n-grams-of-any-n.png", dpi=400)
    plt.show()


# Grab all of the newline-delimited names (with underscores replaced by spaces)
with open("../data/all-raw-names-10.07.20-gen.txt", "r") as all_raw_names:
    names = all_raw_names.readlines()


# Replace newline with dot for standard NLP sentence delimiter, to simplify:
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

    plot_ngram_counts(counts, cutoff_total, ngram_size)

# Find, plot and save the 50 most common ngrams of any size, with removal:
plot_recursive_ngram_counts(
    recursive_get_and_remove_any_n_ngram_counts(names_object, 50))
