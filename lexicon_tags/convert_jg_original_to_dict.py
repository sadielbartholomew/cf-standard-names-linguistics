import pprint
import re

# This original file was taken directly from the hosted location under
# Jonathan Gregory's homespace of his tags for v.14 of the table at:
# 'http://www.met.rdg.ac.uk/~jonathan/CF_metadata/14.1/lexicon'
ORIG_TAG_DIR_RELATIVE = "raw/"
ORIG_TAG_FILE = "jonathan_gregory_table_v14_original_tags.txt"

# NOTE: phrasetype 'tags' are denoted with parentheses surrounding them to
# distinguish them from the phrases themselves, e.g. (species) for e.g. bromine


def get_phrase_to_phrasetype_dicts():
    '''TODO
    '''
    # Grab all the newline-delimited names (w/ underscores replaced by spaces)
    with open(ORIG_TAG_DIR_RELATIVE + ORIG_TAG_FILE, "r") as all_orig_tags:
        tags = all_orig_tags.readlines()

    direct_phrase_to_phrasetype = {}
    compound_phrase_to_phrasetype = {}
    partial_phrase_to_phrasetype = {}

    # All lines (except those untagged) abide by one of these three forms:
    DIRECT_FORM = r"(\w+)\s(\(\w+\))"  # 1->1 i.e. '<phrase> (phrasetype)'
    # 1->partial or 1->many  i.e. '<phrase> <part of phrase>(a phrasetype)'
    COMPOUND_FORM = r"(\w+)\s(\w+\(\w+\))"
    # partial->many, '<part of phrase>(phrasetype) (phrasetype)_(phrasetype)' 
    PARTIAL_FORM = r"(\w+\(\w+\))\s(\(\w+\)_\(\w+\))"
    
    for tagline in tags:
        if "(" not in tagline:  # "untagged"
            # No tag provided, this is the case of the first entries which are
            # stated to be 'untyped' i.e. "not assigned a phrasetype".
            direct_phrase_to_phrasetype[tagline.strip()] = "(untyped)"
        else:
            direct_matches = re.search(DIRECT_FORM, tagline)

            if direct_matches:
                direct_phrase_to_phrasetype[
                    direct_matches.group(1)] = direct_matches.group(2)
            else:
                compound_matches = re.search(COMPOUND_FORM, tagline)
                if compound_matches:
                    compound_phrase_to_phrasetype[
                        compound_matches.group(1)] = compound_matches.group(2)
                else:  # empirically, this works for all lines in orig. file
                    partial_matches = re.search(PARTIAL_FORM, tagline)
                    partial_phrase_to_phrasetype[
                        partial_matches.group(1)] = partial_matches.group(2)

    return (
        direct_phrase_to_phrasetype,
        compound_phrase_to_phrasetype,
        partial_phrase_to_phrasetype,
    )


# Create three separate dicts as the direct/compound/partial distinction may
# be useful for the lexicon analysis, but we might also want to see the
# tags all combined, hence the dict combination below...
d, c, p = get_phrase_to_phrasetype_dicts()
# pprint.pprint(d)  # as an example
all_phrase_to_phrasetypes = {**d, **c, **p}
pprint.pprint(all_phrase_to_phrasetypes)
