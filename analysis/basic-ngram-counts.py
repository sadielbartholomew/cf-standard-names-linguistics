from pprint import pprint
from collections import Counter

from textblob import TextBlob
from textblob import formats

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

SN_DATA_DIR_RELATIVE = "../data/"
SN_DATA_FILE = "all_cf_standard_names_for_table_v79_at_24_08_22.txt"


# Return and plot the most common n-grams, for a given n, up to this number:
CUTOFF_TOTAL = 15
# Highest integer n-gram length where the most common n-gram occurs >1 time
# (simplest to find this manually by inspection of generated graphs):
NGRAM_N_MAX_INTERESTING = 21


def get_standard_names_as_text_blob(name_in_data_dir):
    '''TODO
    '''
    # Grab all the newline-delimited names (w/ underscores replaced by spaces)
    with open(SN_DATA_DIR_RELATIVE + name_in_data_dir, "r") as all_raw_names:
        names = all_raw_names.readlines()

    # Replace newline with dot (standard NLP sentence delimiter), to simplify:
    names = ". ".join(names)
    names = names.replace("\n", "")
    # Create the textblob core object, a TextBlob, of all the names:
    return TextBlob(names)


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

    known_long_name = (
        'tendency of mole concentration of particulate organic matter '
        'expressed as carbon in sea water due to net primary production by'
    )
    validate_known_long_ngram = tuple(known_long_name.split()[:ngram_size])

    # Quick and basic validity check at this point, before plots made etc.:
    if ngram_size <= len(known_long_name.split()):
        assert ngram_counts.get(validate_known_long_ngram, False), (
            "Something is not right as a known n-gram is not being counted.")
    # Also check that n-grams are not being taken across name boundaries:
    bad_sub_ngram = ' '.join(('azimuth', 'angstrom'))
    for ngram_info in ngram_counts:
        ngram_string = ' '.join(ngram_info)
        assert bad_sub_ngram not in ngram_string, (
            'n-grams seem to have been taken across name boundaries which '
            'is not right.'
        )

    if get_x_most_common:
        ngram_counts = dict(ngram_counts.most_common(get_x_most_common))
    else:  # take all, including n-grams only occurring once across the names
        dict(ngram_counts)

    # Format to get phrase e.g. "in air" instead of tuple of words as keys:
    ngram_counts = {
        " ".join(phrase): count for phrase, count in ngram_counts.items()}
    return ngram_counts


def get_and_remove_any_n_ngram_counts_iteration(names):
    '''Return most common n-grams of any n, removing most common when found.
    '''
    most_common_any_n_ngrams = []
    max_count = 0
    # Need to start at 2 this time, else all words will be removed first!
    for size in range(2, NGRAM_N_MAX_INTERESTING):
        # Get n-grams at this size:
        size_n_ngrams = []
        for name in names.sentences:
            ngrams_of_size = name.ngrams(size)
            size_n_ngrams.extend(ngrams_of_size)

        # Find max count at this size and if more or equal to max for any
        # size n, add name(s) to the most_common_any_n_ngrams:
        most_common_size_n = Counter(
            [tuple(wordlist) for wordlist in size_n_ngrams]).most_common(1)
        ngram_max_count = most_common_size_n[0][1]
        if ngram_max_count >= max_count:
            most_common_any_n_ngrams.append(most_common_size_n)
            max_count = ngram_max_count

    # Determine most common name(s) over any size:
    most_common_any_n_ngrams = sorted(
        most_common_any_n_ngrams, key=lambda item: item[0][1], reverse=True)

    # Grab those with the highest count. Note there may be multiple names
    # with the highest so we need to cater for potentially >1 name.
    # TODO: data structure conversion is pretty inefficient here, try to find
    # a better way to retrieve the ngrams with highest count:
    final_most_common_any_n_ngrams = {
        count_object[0][0]: count_object[0][1] for count_object in
        most_common_any_n_ngrams
    }

    highest_counts = max(final_most_common_any_n_ngrams.values())
    overall_most_common = [(ngram, value) for ngram, value in
                           final_most_common_any_n_ngrams.items() if
                           value == highest_counts]

    if len(overall_most_common) > 1:  # see get_and_remove_any_n_ngram_counts
        print(
            "WARNING: there are multiple n-grams with the max. count for this "
            "iteration, namely: '{}'. Will remove and store only the n-grams "
            "in this list which have the highest n in this iteration: ".format(
                "', '". join(
                    list([" ".join(info[0]) for info in overall_most_common])
                )
            ),
            end=''  # print of 'hightest n' n-gram(s) follows below
        )
        max_ngram_length = len(
            max(overall_most_common, key=lambda n: len(n[0]))[0])
        overall_most_common = [ngram_info for ngram_info in
                               overall_most_common if
                               len(ngram_info[0]) == max_ngram_length]
        print(overall_most_common)

    return overall_most_common, names


def get_and_remove_any_n_ngram_counts(names, get_minimum_of_x_most_common):
    '''TODO
    '''
    # Get most common n-gram of any size:
    overall_most_common_ngrams = []
    use_names = names
    while len(overall_most_common_ngrams) < get_minimum_of_x_most_common:
        packed_most_freq_ngrams, all_names = (
            get_and_remove_any_n_ngram_counts_iteration(use_names))
        overall_most_common_ngrams.extend(packed_most_freq_ngrams)

        # Remove this most common n-gram from all names ready to go again:
        reduced_names = []
        for name in all_names.sentences:
            for most_freq_ngram_info in packed_most_freq_ngrams:
                str_ngram = " ".join(most_freq_ngram_info[0])
                if str_ngram in name:
                    name = name.replace(str_ngram, "")
                    name = name.replace("  ", " ")
            reduced_names.append(name)
        reduced_names = "\n".join([str(name) for name in reduced_names])

        # Now use the reduced set of names as input, rather than the originals:
        use_names = TextBlob(reduced_names)

    # Format to get phrase instead of tuple of words as keys:
    ngram_counts = {" ".join(ngram_info[0]): ngram_info[1] for ngram_info in
                    overall_most_common_ngrams}

    return ngram_counts


def set_up_plot_of_ngram_counts(
        ngram_counts, use_colour='thistle'):
    '''TODO
    '''
    plt.rcdefaults()
    fig, ax = plt.subplots()

    names = list(ngram_counts.keys())
    counts = list(ngram_counts.values())

    ax.barh(names, counts, color=use_colour)

    return fig, ax, names, counts


def save_and_show_plot_of_ngram_counts(fig, ax, size, title):
    '''TODO
    '''
    fig.set_size_inches(*size)
    plt.savefig(title, dpi=400)
    plt.show()


def plot_ngram_counts(
        ngram_counts, cutoff_number, ngram_n, threshold_ratio=0.2):
    '''TODO
    '''
    # At n~12, n-grams become so long that they get cut-off by either side of
    # of the canvas, so must effectively centre them by placing inside left:
    put_all_labels_inside_left = False
    if ngram_n > 11:  # point at which (in 10.07.20 data) labels get too long
        put_all_labels_inside_left = True

    fig, ax, names, counts = set_up_plot_of_ngram_counts(ngram_counts)

    plt.axis('tight')
    plt.margins(0, 0.05)

    # `cutoff` is the point at which we switch the name labels to point right
    # and into the graph instead pointing left from the right end of the bar
    # (which looks nicer, but when a bar gets too small relative to the plot
    # width, essentially the longest bar, the name gets cut off to the left.)
    cutoff = threshold_ratio * max(counts)
    # The n-gram names can be rather long, and often get cut off from the left
    # side of the bar plot, so instead (assuming they are not n~12 where so
    # long they are nearly plot width) plot them inside the bars themselves:
    for i, (name, count) in enumerate(ngram_counts.items()):

        # Firstly amend formatting to accommodate very long labels after n~12:
        ha = 'right'
        use_x = count
        if count < cutoff:
            ha = 'left'
        if put_all_labels_inside_left:
            use_x = 0
            ha = 'left'

        label = "{} ({})".format(name, count)
        plt.text(
            s=label, x=use_x, y=i,
            color="indigo", verticalalignment="center",
            horizontalalignment=ha, size=10, wrap=True
        )
    plt.yticks([])

    ax.set_title(
        '{} most common n-grams of size {} for the CF Standard Names'.format(
            cutoff_number, ngram_n)
    )
    ax.set_xlabel('Frequency across all standard names in current table')

    save_and_show_plot_of_ngram_counts(
        fig, ax, (12, 6),
        "../results/ngrams/most-common-{}-grams.png".format(ngram_n)
    )


def plot_recursive_ngram_counts(ngram_counts):
    '''TODO
    '''
    fig, ax, names, counts = set_up_plot_of_ngram_counts(
        ngram_counts, use_colour=['lightgreen', 'lightsteelblue'])

    # Axes formatting:
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(which='major', length=12)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", direction="in", pad=-10)
    plt.setp(ax.get_yticklabels(), ha="left")

    for index, (name, count) in enumerate(ngram_counts.items()):
        # Match colours as roughly darker versions of alternating bar colours:
        colour = 'forestgreen'
        ha = 'left'
        if index % 2:
            colour = 'midnightblue'
            ha = 'right'

        # Prevent leftmost and rightmost name labels from being cut off canvas
        if count < 45:
            ha = 'left'
        elif count > 1000:
            ha = 'right'

        # Add some padding between data bar and text label
        use_x = count
        pad_factor = 0.05
        if ha == 'left':
            use_x *= 1 + pad_factor
        else:
            use_x *= 1 - pad_factor

        label = "{} ({})".format(name, count)
        plt.text(
            s=label, x=use_x, y=index,
            color=colour, verticalalignment="center",
            ha=ha, size=11,
        )
    plt.yticks([])

    # Labelling:
    ax.set_title(
        '{} most common n-grams of any size for the '
        'CF Standard Names'.format(len(names))
    )
    ax.text(
        0.7, 0.95,
        "\n".join((
            'Note the colours of the bars',
            'and their labels do not have',
            'any meaning; they are plotted',
            'in alternating colours to make it',
            'easier to distinguish neighbouring',
            'bars and their labels.',
        )),
        transform=ax.transAxes, verticalalignment='top', size=11
    )
    ax.set_xlabel(
        'Frequency across all standard names in current table (log scale)')

    save_and_show_plot_of_ngram_counts(
        fig, ax, (12, 12),
        "../results/ngrams/recursive-n-grams-of-any-n.png"
    )


def plot_most_common_ngrams_without_removal(
        min_n, max_n, cutoff, display=True):
    """Return, plot and save the most common n-grams from min_n to max_n n."""
    for ngram_size in range(min_n, max_n + 1):
        # Get counts and conduct an assertion check
        counts = get_ngram_counts(names_blob, ngram_size, cutoff)
        if display:
            pprint(counts)
        plot_ngram_counts(counts, cutoff, ngram_size)


def plot_most_common_ngrams_with_removal(cutoff, display=True):
    """Return, plot and save the most common n-grams of any n with removal."""
    any_n_most_common = get_and_remove_any_n_ngram_counts(names_blob, cutoff)
    if display:
        pprint(any_n_most_common)
    plot_recursive_ngram_counts(any_n_most_common)


if __name__ == "__main__":
    names_blob = get_standard_names_as_text_blob(SN_DATA_FILE)
    plot_most_common_ngrams_without_removal(
        1, NGRAM_N_MAX_INTERESTING, CUTOFF_TOTAL)
    plot_most_common_ngrams_with_removal(60)
