from collections import Counter
import json
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from textblob import TextBlob
from textblob import formats


SHORT_CIRCUIT_TO_N_NAMES = False  #1000  # int to short circuit, or False to not
FREQUENCY_CUTOFF = 50  # v >= FREQUENCY_CUTOFF for inclusion
# samples to do: 1, 2, 3, 5, 10, 15, 25, 50, 100, 250, 500 

SN_DATA_DIR_RELATIVE = "../data/"
SN_DATA_FILE = "all_cf_standard_names_for_table_v83_at_30_11_23.txt"
SN_DATA_IN_USE = SN_DATA_DIR_RELATIVE + SN_DATA_FILE

SAVE_DIR = "calculated_data_to_persist"

# Filename or False to re-generate the data
USE_PREV_DATA = f"{SAVE_DIR}/all_ngram_counts_with_cutoff_25.json"
# Note: ~21,000 nodes for the 2 cutoff full-dataset graph!


# ---------------------------------------------------------------------------
# DATA CREATION -------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_standard_names_as_text():
    """Take the names input data from file and convert format."""
    # Open and read the file with all table standard names
    with open(SN_DATA_IN_USE, "r") as all_raw_names:
        names = all_raw_names.readlines()

    return names


def define_node_input_data():
    """Define and return *all* data to use as input to the graph."""
    names = get_standard_names_as_text()
    # For development, use a small subset of the data to reduce script run time
    if SHORT_CIRCUIT_TO_N_NAMES:
        names = names[:SHORT_CIRCUIT_TO_N_NAMES]

    return names


def find_min_and_max_name_length(names):
    """Find the minimum- and maximum-length name(s) by word count."""

    # Easiest to find length in words via number of spaces in name plus 1
    length_in_words = np.array([name.count(" ") + 1 for name in names])

    ### print("USING FOR MIN/MAX LEN BY WORD\n", names)

    wordlen_min = min(length_in_words)
    wordlen_max = max(length_in_words)
    wordlen_min_names = list(
        [
            names[index] for index in
            np.where(length_in_words == wordlen_min)[0]
        ]
    )
    wordlen_max_names = list(
        [
            names[index] for index in
            np.where(length_in_words == wordlen_max)[0]
        ]
    )

    range_of_wordlens = {
        wordlen_min: wordlen_min_names, wordlen_max: wordlen_max_names}
    ### pprint(range_of_wordlens)

    return range_of_wordlens


def get_range_of_n_to_check(word_length_ranges):
    """Return the range of size n to check for n-grams of, as an iterator."""
    _, max_n = word_length_ranges.keys()
    max_n += 1  # since will input this to range, want to capture max wordlen
    return range(1, max_n)


def get_ngram_counts(names, ngram_size, cutoff_freq=1):
    """Find all n-grams of given n across the names, with their counts.

    Note: borrows heavily from 'get_ngram_counts' in 'basic-ngram-counts'
    script. TODO: consolidate these into one function to use in both.
    """
    names = TextBlob(names)

    # Note, can't simply do:
    # ngram = names.ngrams(ngram_size)
    # as it takes ngrams across sentences, i.e. start & end of names in order.
    names_sentences = names.sentences
    ngrams = []
    for name in names_sentences:
        ngrams.extend(name.ngrams(ngram_size))

    ngram_counts = Counter(
        [tuple(wordlist) for wordlist in ngrams])  # need tuple for hashability
    ### print("FINAL NGRAM COUNTS ARE:\n", ngram_counts)

    # (ONLY HELPS WITH FULL DATA SET VALIDATION - PASSES HARMLESSLY OTHERWISE!)
    # Also check that n-grams are not being taken across name boundaries:
    bad_sub_ngram = ' '.join(('azimuth', 'angstrom'))
    for ngram_info in ngram_counts:
        ngram_string = ' '.join(ngram_info)
        assert bad_sub_ngram not in ngram_string, (
            'n-grams seem to have been taken across name boundaries which '
            'is not right.'
        )

    # Format to get phrase e.g. "in air" instead of tuple of words as keys:
    ngram_counts = {
        " ".join(phrase): count for phrase, count in ngram_counts.items()}

    # Remove anything only ocuring once and not reecurring
    ngram_counts = {
        k: v for k, v in ngram_counts.items() if v >= cutoff_freq
    }

    return ngram_counts


def get_all_ngram_counts(init_data):
    """TODO."""
    space_and_dot_delim_data = ". ".join(init_data)
    data = space_and_dot_delim_data.replace("\n", "")
    print("DATA TO USE FOR N-GRAMS CALC IS...")
    pprint(data)

    all_ngram_data = {}
    for n_size in range_n:
        ### print("CALC'ING", n_size)
        counts = get_ngram_counts(data, n_size, FREQUENCY_CUTOFF)
        # Filter out all n-gram keys where the dict is empty due to no n-grams
        if counts:
            all_ngram_data[n_size] = counts

    return all_ngram_data


# ----------------------------------------------------------------------------
# DATA REFORMATTING FOR GRAPH ------------------------------------------------
# ----------------------------------------------------------------------------

def reformat_nodes_data(json_ngram_data):
    """Take JSON of n-gram counts and convert it to node and edge form."""

    all_nodes_keyed_by_id = {}
    # 1. Flatten dict so all n-grams become values of a unique identified int
    node_id = 1
    for ngram_set in json_ngram_data.values():
        for ngram, ngram_count in ngram_set.items():
            all_nodes_keyed_by_id[node_id] = (ngram, ngram_count)
            node_id += 1

    return all_nodes_keyed_by_id

# ----------------------------------------------------------------------------
# GRAPH CREATION -------------------------------------------------------------
# ----------------------------------------------------------------------------

def add_nodes_to_graph(graph, inputs_for_nodes):
    """Add nodes to the graph."""
    ### print("NODES ADDED\n", inputs_for_nodes)
    graph.add_nodes_from(inputs_for_nodes)


def create_edge_spec(nodes_added):
    """Define the data to encode the edges required to connect nodes."""
    return []


def add_edges_to_connect_nodes_in_graph(graph, edge_spec):
    """Add edges to connect the nodes in the graph."""
    graph.add_edges_from(edge_spec)


def define_node_labels(nodes):
    """Define data for the labels to be applied on the graph nodes/edges."""
    labels_mapping = {k: v[0] for k, v in nodes.items()}

    return labels_mapping


def sizes_for_nodes(graph, nodes):
    """TODO."""
    scale_factor = 5  # make smaller by this amount so nodes aren't huge
    nodes_size_mapping = [v[1]/scale_factor for v in nodes.values()]
    return nodes_size_mapping


def label_graph(graph, labels):
    """Add labelling onto the graph."""
    # Need to return and use graph object so re-labelling is applied - why?
    return nx.relabel_nodes(graph, labels)


def post_processing_of_graph(graph, layout):
    """Actions to apply to modify the graph after layout specification."""
    # Offset the node labels so they are above the nodes and don't cover them
    offset = 0.00
    label_pos_vals = [(x, y + offset) for x, y in layout.values()]
    label_pos = dict(zip(layout.keys(), label_pos_vals))
    ### print("LABEL POS IS\n", label_pos)

    nx.draw_networkx_labels(graph, label_pos, font_size=6, alpha=0.7)


def draw_graph_with_layout(graph, node_sizes, node_colours):
    """Apply a layout and customisations to define how to draw the graph."""
    options = {
        ###"node_color": "tab:red",
        ###"node_size": 5,
        "edge_color": "tab:blue",
        "alpha": 0.5,
        "with_labels": False,  # labels are set on later, False to avoid dupes
        "font_size": 4,
    }

    # Specify graph layout here! E.g. 'spiral_', 'spring_', 'shell_', etc.
    layout = nx.shell_layout(graph)
    nx.draw(
        graph, layout, node_size=node_sizes,
        node_color=node_colours, cmap=plt.cm.hsv,
        **options
    )

    return graph, layout


def create_sn_nrgam_graph(nodes):
    """Create a directed graph of n-gram links across the CF Standard Names."""
    print("GRAPH RAW DATA TO WORK WITH IS:")
    pprint(nodes)

    # 0. Initiate graph
    G = nx.DiGraph()  # Directed graphs with self loops

    # 1. Add nodes without links (edges) to graph
    add_nodes_to_graph(G, nodes)

    # 2. Generate edges to add
    ###edges = create_edge_spec(edges)
    # 3. Add those edges
    ###add_edges_to_connect_nodes_in_graph(G, edges)

    # 4. Generate labels to apply on the nodes and/or edges
    labels = define_node_labels(nodes)
    # 5. Add those labels
    G = label_graph(G, labels)

    # 6. Set-up node size proportional to the frequency of occurence.
    node_sizes = sizes_for_nodes(G, nodes)

    # 7. Colour nodes by n-gram size. It is simpler just to re-calculate
    # the length based on the actual n-gram/name label rather than query
    # it from the nodes data dict.
    print("LABELS ARE:\n", labels)
    node_colours = [l.count(" ") + 1 for l in labels.values()]

    # N-2. <Numbering>

    # N-1. Customise the layout to use and plot the graph using it
    G, layout = draw_graph_with_layout(
        G, node_sizes, node_colours)

    # N. Apply post-processing, e.g. to reposition node labels based on layout
    post_processing_of_graph(G, layout)

    plt.show()


# ----------------------------------------------------------------------------
# ALTOGETHER -----------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # 0. Define data
    # OR (ALTNERATIVE LATER) 3. Load from file to save re-generating
    if USE_PREV_DATA:
        with open(USE_PREV_DATA, "r") as data_input_file:
            all_ngram_data = json.load(data_input_file)
    else:
        orig_data = define_node_input_data()

        # 1. Find the minimum and maximum n-gram length i.e. length by word count
        word_length_ranges = find_min_and_max_name_length(orig_data)
        # 2. Use this range of n to set n-gram limits to find and intialise
        #    data structure to use. Values are set to None until calculation.
        range_n = get_range_of_n_to_check(word_length_ranges)

        # 3. Find all n-grams occuring more than the cutoff amount across the
        #    names for all given n in the word_length_ranges range
        all_ngram_data = get_all_ngram_counts(orig_data)

    print("NGRAM DATA STRUCTURE CALC'ED IS:")
    pprint(all_ngram_data)

    # 4. Store the data to avoid re-generating
    if not USE_PREV_DATA and not SHORT_CIRCUIT_TO_N_NAMES:
        filename_to_write = (
            f"{SAVE_DIR}/"
            f"all_ngram_counts_with_cutoff_{FREQUENCY_CUTOFF}.json"
        )
        with open(filename_to_write, "w") as f:
            json.dump(all_ngram_data, f)

    # 4. Interface to data structure holding nodes and edge info.
    ngram_data_nodes = reformat_nodes_data(all_ngram_data)
    
    # N. Finally, plot the network graph!
    create_sn_nrgam_graph(ngram_data_nodes)  # No edges for now!
