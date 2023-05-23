#!/usr/bin/bash

for EXPERIMENT in "bills_broad" "wikitext_broad" "wikitext_specific"; do
    # prevent from showing
    DISPLAY="" python3 src_vilem/03-figures_nmpi_llm.py --experiment ${EXPERIMENT}
done