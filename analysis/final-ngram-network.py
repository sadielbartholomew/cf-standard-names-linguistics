import networkx as nx
import matplotlib.pyplot as plt


SHORT_CIRCUIT_TO_N_NAMES = 50  # int to short circuit, or False to not

SN_DATA_DIR_RELATIVE = "../data/"
SN_DATA_FILE = "all_cf_standard_names_for_table_v83_at_30_11_23.txt"


# ---------------------------------------------------------------------------
# DATA CREATION -------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_standard_names_as_text():
    """Take the names input data from file and convert format."""
    # Open and read the file with all table standard names
    with open(SN_DATA_DIR_RELATIVE + SN_DATA_FILE, "r") as all_raw_names:
        names = all_raw_names.readlines()

    formatted_names = [name.strip("\n") for name in names]
    return formatted_names


def define_node_input_data():
    """Define and return *all* data to use as input to the graph."""
    names = get_standard_names_as_text()

    # For development, use a small subset of the data to reduce script run time
    if SHORT_CIRCUIT_TO_N_NAMES:
        names = names[:SHORT_CIRCUIT_TO_N_NAMES]

    return names


# ----------------------------------------------------------------------------
# GRAPH CREATION -------------------------------------------------------------
# ----------------------------------------------------------------------------

def add_nodes_to_graph(graph, inputs_for_nodes):
    """Add nodes to the graph."""
    print("NODES ADDED\n", inputs_for_nodes)
    graph.add_nodes_from(inputs_for_nodes)


def create_edge_spec(nodes_added):
    """Define the data to encode the edges required to connect nodes."""
    # DUMMY CONNECTIONS FOR NOW!
    edge_to_edge_dummy = {n: n + 1 for n in range(1, len(nodes_added) - 1)}
    print("DUMMY IS\n", edge_to_edge_dummy)

    get_edges = [
        (nodes_added[k], nodes_added[v]) for k, v in edge_to_edge_dummy.items()
    ]
    print("EDGE LIST IS\n", get_edges)
    return get_edges


def add_edges_to_connect_nodes_in_graph(graph, edge_spec):
    """Add edges to connect the nodes in the graph."""
    graph.add_edges_from(edge_spec)


def define_node_labels(nodes):
    """Define data for the labels to be applied on the graph nodes/edges."""
    indices = range(1, len(nodes) + 1)

    # MUST TAKE CARE: IDENTICAL LABELS WILL COMBINE THOSE NODES TO ONE!
    labels = []
    for index, n in enumerate(nodes):
        get_first_letters_words = [f[0].upper() for f in n.split(" ")]
        # CRUCIAL TO UE INDEX IN CASE OF DUPLICATES IN NAME!
        labels.append(str(index) + ": " + "".join(get_first_letters_words))

    short_labels_for_nodes = {key: value for key, value in zip(nodes, labels)}
    return short_labels_for_nodes


def label_graph(graph, labels):
    """Add labelling onto the graph."""
    # Need to return and use graph object so re-labelling is applied - why?
    return nx.relabel_nodes(graph, labels)


def post_processing_of_graph(graph, layout):
    """Actions to apply to modify the graph after layout specification."""
    # Offset the node labels so they are above the nodes and don't cover them
    offset = 0.02
    label_pos_vals = [(x, y + offset) for x, y in layout.values()]
    label_pos = dict(zip(layout.keys(), label_pos_vals))
    print("LABEL POS IS\n", label_pos)

    nx.draw_networkx_labels(graph, label_pos, font_size=6, alpha=0.7)


def draw_graph_with_layout(graph):
    """Apply a layout and customisations to define how to draw the graph."""
    options = {
        "node_color": "tab:red",
        "node_size": 5,
        "edge_color": "tab:blue",
        "alpha": 0.7,
        "with_labels": False,  # labels are set on later, False to avoid dupes
        "font_size": 6,
    }

    # Specify graph layout here! E.g. 'spiral_', 'spring_', 'shell_', etc.
    layout = nx.shell_layout(graph)
    nx.draw(graph, layout, **options)

    return graph, layout


def create_sn_nrgam_graph(NAMES_DATA):
    """Create a directed graph of n-gram links across the CF Standard Names."""
    # DATA - DEFINE CURRENT INPUTS:
    nodes = NAMES_DATA

    # 0. Initiate graph
    G = nx.DiGraph()  # Directed graphs with self loops

    # 1. Add nodes without links (edges) to graph
    add_nodes_to_graph(G, nodes)

    # 2. Generate edges to add
    edges = create_edge_spec(nodes)
    # 3. Add those edges
    add_edges_to_connect_nodes_in_graph(G, edges)

    # 4. Generate labels to apply on the nodes and/or edges
    labels = define_node_labels(nodes)
    # 5. Add those labels
    G = label_graph(G, labels)

    # 6. Customise the layout to use and plot the graph using it
    G, layout = draw_graph_with_layout(G)

    # 7. Apply post-processing, e.g. to reposition node labels based on layout
    post_processing_of_graph(G, layout)

    # END


# ----------------------------------------------------------------------------
# ALTOGETHER -----------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # 0. Define data
    data = define_node_input_data()

    # 1. Finally, plot the network graph!
    create_sn_nrgam_graph(data)
    plt.show()
