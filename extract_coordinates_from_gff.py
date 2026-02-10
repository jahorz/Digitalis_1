#!/usr/bin/env python3
import sys

busco_file = "PATH/TO/busco_ids.txt"
gff_file = "PATH/TO/genome.anno.gff"
out_file = "PATH/TO/busco_genes.bed"

# Load BUSCO IDs
busco_ids = set()
with open(busco_file) as f:
    for line in f:
        busco_ids.add(line.strip())

with open(gff_file) as gff, open(out_file, "w") as out:
    for line in gff:
        if line.startswith("#"):
            continue
        fields = line.rstrip().split("\t")
        if len(fields) != 9:
            continue

        seqid, source, feature, start, end, score, strand, phase, attrs = fields

        if feature != "mRNA":
            continue

        # Parse attributes
        attr_dict = {}
        for item in attrs.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                attr_dict[k] = v

        gene_id = attr_dict.get("ID")
        if gene_id in busco_ids:
            # BED is 0-based start
            out.write(f"{seqid}\t{int(start)-1}\t{end}\t{gene_id}\n")

print(f"Done. Output written to {out_file}")
