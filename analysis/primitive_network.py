import networkx as nx
import matplotlib.pyplot as plt

SN_DATA_DIR_RELATIVE = "../data/"
SN_DATA_FILE = "all_cf_standard_names_for_table_v83_at_30_11_23.txt"


def get_standard_names_as_text(name_in_data_dir):
    # Grab all the newline-delimited names (w/ underscores replaced by spaces)
    with open(SN_DATA_DIR_RELATIVE + name_in_data_dir, "r") as all_raw_names:
        names = all_raw_names.readlines()

    return [name.strip("\n") for name in names]


def create_sn_graph(names):
    # MAKE NODES, ETC.
    G = nx.DiGraph()  # Directed graphs with self loops
    ###for name in names:

    # NODES:
    G.add_nodes_from(names)

    # EDGES:
    # dummy ones for now!
    edge_to_edge_dummy = {n: n + 1 for n in range(1, len(names) - 1)}
    print("DUMMY IS", edge_to_edge_dummy)
    get_edges = [(names[k], names[v]) for k, v in edge_to_edge_dummy.items()]
    print("EDGE LIST IS", get_edges)
    G.add_edges_from(get_edges)

    indices = range(1, len(names) + 1)

    # SINCE MUST TAKE CARE: IF LABELS ARE IDENTICAL, WILL COMBINE THOSE NODES
    # TO ONE!
    labels = []
    for index, n in enumerate(names):
        get_first_letters_words = [f[0].upper() for f in n.split(" ")]
        # CRUCIAL TO UE INDEX IN CASE OF DUPLICATES IN NAME!
        labels.append(str(index) + ": " + "".join(get_first_letters_words))

    short_labels_for_names = {key: value for key, value in zip(names, labels)}
    G = nx.relabel_nodes(G, short_labels_for_names)

    ###pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    ###nx.kamada_kawai_layout(G)  # how to position the nodes
    options = {
        "node_color": "tab:red",
        "node_size": 60,
        "edge_color": "tab:blue",
        "alpha": 0.7,
        "with_labels": False,  # labels are set on later, False to avoid dups
        "font_size": 6,
    }
    pos = nx.spiral_layout(G)
    nx.draw(G, pos, **options)

    # Offset the node labels so they are above the nodes so don't cover them
    offset = 0.02
    label_pos_vals = [(x, y + offset) for x, y in pos.values()]
    label_pos = dict(zip(pos.keys(), label_pos_vals))
    print("LABEL POS IS", label_pos)
    nx.draw_networkx_labels(G, label_pos, font_size=6, alpha=0.7)

    plt.show()


# MAIN:
# NOTE: ONLY TAKING FIRST 200 FOR SPEED, FOR NOW - adapt for all or for less:
names = get_standard_names_as_text(SN_DATA_FILE)[:20]
create_sn_graph(names)
