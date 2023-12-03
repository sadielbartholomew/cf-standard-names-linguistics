from collections import Counter
from itertools import pairwise  # Needs Py 3.10!
import json
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from textblob import TextBlob
from textblob import formats


SHORT_CIRCUIT_TO_N_NAMES = False  #1000  # int to short circuit, or False to not


# Note: ~21,000 nodes for the 2 cutoff full-dataset graph! Graphs are big!
# CUTOFF OF 3 OR LESS IS TOO MEMORY-INTENSIVE! NEED TO USE JOB(?) ARCHER?
FREQUENCY_CUTOFF = 250  # v >= FREQUENCY_CUTOFF for inclusion
# samples to do: 1, 2, 3, 5, 10, 15, 25, 50, 100, 250, 500
# NEW ONES: 300 - for spiral graph to add?

SN_DATA_DIR_RELATIVE = "../data/"
SN_DATA_FILE = "all_cf_standard_names_for_table_v83_at_30_11_23.txt"
SN_DATA_IN_USE = SN_DATA_DIR_RELATIVE + SN_DATA_FILE

SAVE_DIR = "calculated_data_to_persist"
SAVE_DIR_PLOTS = "raw_plots"

# Filename or False to re-generate the data
USE_PREV_DATA = (
    f"{SAVE_DIR}/all_ngram_counts_with_cutoff_{FREQUENCY_CUTOFF}.json"
)


# Quick graph customisation
LABELS_ON = True
LABEL_OFFSET = 0.01  # 0.02-0.05 is good

# Use rainbow colours - need lack of white tones or repeated tones at each
# end of the spectrum, and strong bright variation in colour.
CMAP = plt.cm.rainbow  # tubro, rainbow, gist_rainbow, jet
SAVE_DPI = 750  # publication needs 600 or above


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


def reverse_nodes_id_dict(all_nodes_keyed_by_id):
    """TODO.

    TODO: shouldn't be necessary with a better data structure approach.
    Reconsider approach to data wrangling later...
    """
    return {v[0]: k for k, v in all_nodes_keyed_by_id.items()}


def generate_edges(json_ngram_data, all_nodes_keyed_by_id):
    """TODO.

    Note: probably performance heavy.
    """
    node_id_lookups = reverse_nodes_id_dict(all_nodes_keyed_by_id)

    links = []
    for n_size, n_grams in json_ngram_data.items():
        # Get the n-gram and (n-1)-gram data to compare to find links for edges
        n_size = int(n_size)  # TODO: why has this become a string?
        n_one_less = n_size - 1
        if n_size >= 1 and str(n_one_less) in json_ngram_data:  # else pass
            n_grams_for_n_one_less = json_ngram_data[str(n_one_less)]
        else:
            continue

        # Find all (n-1)-grams contained in any n-grams to add as edges
        for smaller_ngram in n_grams_for_n_one_less.keys():
            for larger_ngram in n_grams.keys():
                if smaller_ngram in larger_ngram:
                    print(
                        "EDGE MATCH AT:\n", larger_ngram, "AND", smaller_ngram)
                    links.append((smaller_ngram, larger_ngram))

    print("FINAL LINKS LIST IS:\n")
    pprint(links)

    if LABELS_ON:
        return links
    else:
        # Now convert the links in 2-tuples of names to their node IDs ready to
        # specifiy for the graph.
        node_id_links = []
        for link in links:
            smaller_ngram, larger_ngram = link
            node_id_links.append(
                (node_id_lookups[smaller_ngram], node_id_lookups[larger_ngram]))

        print("FINAL NODE ID LINKS LIST IS:\n")
        pprint(node_id_links)

        return node_id_links

# ----------------------------------------------------------------------------
# GRAPH CREATION -------------------------------------------------------------
# ----------------------------------------------------------------------------

def generate_dummy_egdes(json_ngram_data, all_nodes_keyed_by_id):
    """Create false edges not to be displayed to force a better layout."""
    # Only dummy egdes for now are to connect all of the n-gram central
    # n-grams together in a circle - hence need a cyclic link system
    # of (0, 1), (1, 2) ..., (N, 0).
    dummy_links = list(pairwise(json_ngram_data["1"]))
    dummy_links.append((dummy_links[0][0], dummy_links[-1][-1]))
    print("DUMMY LINKS ARE:\n", list(dummy_links))

    # TODO create helper function for this logic (if LABEL_IN switch as above)
    if LABELS_ON:
        return dummy_links
    else:
        # Now convert the links in 2-tuples of names to their node IDs ready to
        # specifiy for the graph.
        node_id_links = []
        for link in links:
            link_start, link_end = link
            node_id_links.append(
                (node_id_lookups[link_start], node_id_lookups[link_end]))

    return dummy_links


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
    # Make larger by this amount, Usually will want it <1 to reduce size.
    scale_factor = 1/2
    nodes_size_mapping = [v[1] * scale_factor for v in nodes.values()]
    return nodes_size_mapping


def label_graph(graph, labels):
    """Add labelling onto the graph."""
    # Need to return and use graph object so re-labelling is applied - why?
    return nx.relabel_nodes(graph, labels)


def post_processing_of_graph(graph, layout):
    """Actions to apply to modify the graph after layout specification."""
    # Offset the node labels so they are above the nodes and don't cover them
    if LABELS_ON:
        offset = LABEL_OFFSET
        label_pos_vals = [(x, y + offset) for x, y in layout.values()]
        label_pos = dict(zip(layout.keys(), label_pos_vals))
        ### print("LABEL POS IS\n", label_pos)

        nx.draw_networkx_labels(graph, label_pos, font_size=6, alpha=0.8)


def draw_graph_with_layout(graph, node_sizes, node_colours, edge_colours):
    """Apply a layout and customisations to define how to draw the graph."""
    options = {
        ###"node_color": "tab:red",
        ###"node_size": 5,
        ### "edge_color": "tab:gray",
        "alpha": 0.6,
        "with_labels": False,  # labels are set on later, False to avoid dupes
        "font_size": 4,
        "linewidths": 1,  # adds a border to the node colours!
        "edgecolors": "black",  # this is the colour of the node border!
    }

    # Specify graph layout here!
    # SEE ALL LAYOUT OPTIONS AT:
    # https://networkx.org/documentation/stable/reference/...
    # ...drawing.html#module-networkx.drawing.layout
    #
    # Good options seem to be:
    # spiral (probs best to show whole structure of nodes, but NOT links)
    # kamada_kawai_layout
    #
    # (not shell, all on a circle, though that is good to see all connections)
    # (not spring, is quite random) (not spectral, gives weird clustering)
    # (shell is good for showing links between)

    # ONE OPTION: MULTIPARTITE IN LAYERS OF N-GRAM INCREASING SIZE?

    # NOT A TREE SO WON'T WORK!:
    ###layout = nx.nx_agraph.graphviz_layout(graph, prog="twopi", root=0)
    layout = nx.shell_layout(graph)
    ###layout = nx.kamada_kawai_layout(graph)

    nx.draw(
        graph, layout, node_size=node_sizes,
        node_color=node_colours, edge_color=edge_colours,
        cmap=CMAP, edge_cmap=CMAP,
        **options
    )

    return graph, layout


def create_sn_nrgam_graph(nodes, edges):
    """Create a directed graph of n-gram links across the CF Standard Names."""
    print("GRAPH RAW DATA TO WORK WITH IS:")
    pprint(nodes)

    # 0. Initiate graph
    G = nx.DiGraph()  # Directed graphs with self loops

    # 1. Add nodes without links (edges) to graph
    add_nodes_to_graph(G, nodes)

    # Create names as node labels
    labels = define_node_labels(nodes)

    # 4. Generate labels to apply on the nodes and/or edges
    if LABELS_ON:
        # 5. Add those labels
        G = label_graph(G, labels)

    # 6. Set-up node size proportional to the frequency of occurence.
    node_sizes = sizes_for_nodes(G, nodes)

    # 7. Colour nodes by n-gram size. It is simpler just to re-calculate
    # the length based on the actual n-gram/name label rather than query
    # it from the nodes data dict.
    print("LABELS ARE:\n", labels)
    node_colours = [l.count(" ") + 1 for l in labels.values()]
    

    # N. Add those edges
    add_edges_to_connect_nodes_in_graph(G, edges)
    # Use the same colours for the edges as the node pointing to:
    if LABELS_ON:
        edge_colours = [e[1].count(" ") + 1 for e in edges]
    else:
        edge_colours = [nodes[e[1]][0].count(" ") + 1 for e in edges]
    ### print("EDGE COLOURS ARE:", edge_colours)

    # N-2. <Numbering>

    # N-1. Customise the layout to use and plot the graph using it
    G, layout = draw_graph_with_layout(
        G, node_sizes, node_colours, edge_colours)

    # N. Apply post-processing, e.g. to reposition node labels based on layout
    post_processing_of_graph(G, layout)

    # Add a legend
    # TODO.


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
    ngram_data_edges = generate_edges(all_ngram_data, ngram_data_nodes)
    ### ngram_data_edges += generate_dummy_egdes(all_ngram_data, ngram_data_nodes)

    # N. Finally, plot the network graph!
    create_sn_nrgam_graph(ngram_data_nodes, ngram_data_edges)

    # Finally save and show the plot. Save per cutoff and label on/off to get
    # variety of plots to compare.
    plt.savefig(
        f"{SAVE_DIR_PLOTS}/"
        f"digraph_cutoff{FREQUENCY_CUTOFF}_labels{int(LABELS_ON)}_shell.png",
        dpi=SAVE_DPI,
    )
    plt.show()
