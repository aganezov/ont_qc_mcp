# Public ONT sequencing fixture

The bundled `hg002_ont_chr1.*` files are a small subset of **HG002** from
[Oxford Nanopore's May 2023 GIAB release](https://epi2me.nanoporetech.com/giab-2023.05/).
The release used PromethION, LSK114 and 5 kHz sampling. This fixture uses the
`hg002_sup_60x` analysis, aligned to
`GCA_000001405.15_GRCh38_no_alt_analysis_set.fna`.

| File | Contents |
| --- | --- |
| `hg002_ont_chr1.bam` / `.bam.bai` | 69 alignments overlapping `chr1:4130001-4140000` (one-based inclusive) |
| `hg002_ont_chr1.fq.gz` | 60 primary-record sequences, totaling 1,701,648 bases, derived from that BAM |
| `hg002_ont_chr1.vcf.gz` / `.vcf.gz.tbi` | 27 matching-region Clair3 1.0.0 calls from the same HG002 analysis |

These are ONT-derived variant **calls**, not a GIAB truth set. The BAM retains
complete records, so reads and alignments can extend beyond the selected 10 kb
interval. Secondary and supplementary alignments are excluded only when deriving
FASTQ (`samtools fastq -F 0x900`). The BAM retains its full reference dictionary;
whole-reference QC therefore includes reference bases with no reads in this subset.
Use the selected interval for meaningful local coverage examples.

The data files total about 3.4 MB. Synthetic fixtures remain separate known-answer
controls. This subset is for small workflow checks, not a representative
whole-genome performance or accuracy benchmark.

## Attribution and provenance

Oxford Nanopore Technologies Benchmark Datasets, accessed September 19, 2026,
from https://registry.opendata.aws/ont-open-data/.
The source data and these derived sequencing fixtures are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), separately from the
repository's MIT-licensed software. HG002 (GM24385 / NA24385) is a public reference
sample from the NIGMS Human Genetic Cell Repository at Coriell.

[`provenance.json`](provenance.json) records exact source object URLs, sizes,
ETags, tool versions, selection rules and SHA-256 hashes for each bundled data
file and captured native output. S3 ETags are source identifiers, not SHA-256
checksums. No biological records or alignment coordinates were edited. To keep
headers usable, 23,303 upstream program-history records were removed;
the original `ID:minimap2` record with `-ax map-ont` and all reference records
remain. A comment identifies the fixture transformation. The alignment SAM
records were byte-identical before and after reheadering. The manifest records
both the original header hash and the alignment-record hash.

The `raw/nanoq_hg002_ont_chr1.json` and `raw/cramino_hg002_ont_chr1.json` captures
come from these files using nanoq 0.10.0 and cramino 1.4.1. Cramino's file creation
time is capture metadata and can change on regeneration.

## Recreate the subset

Use samtools/bcftools 1.24 from the documented toolchain. Run from the repository
root, with an empty scratch directory. The commands fetch the indexes (about
67 MB combined) and use HTTP range requests for the data; they do not download
the 205 GB source BAM. Regeneration requires network access; ordinary tests use
the bundled files offline. Keep the new outputs in scratch until verified.

```sh
export FIXTURE_BUILD="$(mktemp -d)"
ONT_SOURCE=https://ont-open-data.s3.eu-west-1.amazonaws.com/giab_2023.05/analysis/variant_calling/hg002_sup_60x
curl --fail --location "$ONT_SOURCE/hg002.haplotagged.bam.bai" -o "$FIXTURE_BUILD/source.bam.bai"
curl --fail --location "$ONT_SOURCE/hg002.wf_snp.vcf.gz.tbi" -o "$FIXTURE_BUILD/source.vcf.gz.tbi"
samtools view --no-PG -b -o "$FIXTURE_BUILD/source.bam" \
  "$ONT_SOURCE/hg002.haplotagged.bam##idx##$FIXTURE_BUILD/source.bam.bai" chr1:4130001-4140000
bcftools view --no-version -Oz -o "$FIXTURE_BUILD/hg002_ont_chr1.vcf.gz" \
  -r chr1:4130001-4140000 "$ONT_SOURCE/hg002.wf_snp.vcf.gz##idx##$FIXTURE_BUILD/source.vcf.gz.tbi"
samtools view --no-PG -H "$FIXTURE_BUILD/source.bam" > "$FIXTURE_BUILD/source.header.sam"
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['FIXTURE_BUILD'])
lines = (root / 'source.header.sam').read_text().splitlines()
kept = [line for line in lines if not line.startswith('@PG\t')]
kept.append(next(line for line in lines if line.startswith('@PG\t') and '\tID:minimap2\t' in line))
kept.append('@CO\tHG002 ONT regional fixture; redundant upstream program history omitted; see provenance.json')
(root / 'fixture.header.sam').write_text('\n'.join(kept) + '\n')
PY
samtools reheader -P "$FIXTURE_BUILD/fixture.header.sam" "$FIXTURE_BUILD/source.bam" > "$FIXTURE_BUILD/hg002_ont_chr1.bam"
samtools index "$FIXTURE_BUILD/hg002_ont_chr1.bam"
bcftools index --tbi "$FIXTURE_BUILD/hg002_ont_chr1.vcf.gz"
samtools fastq -n -F 0x900 "$FIXTURE_BUILD/hg002_ont_chr1.bam" | gzip -n > "$FIXTURE_BUILD/hg002_ont_chr1.fq.gz"
```

After copying verified outputs into this directory, refresh captures and manifest
hashes when intentionally regenerating the fixture:

```sh
nanoq --stats --json --input tests/fixtures/real/hg002_ont_chr1.fq.gz
cramino --format json tests/fixtures/real/hg002_ont_chr1.bam
```

The separate `fetch_real_data.py` helper is not this fixture's regeneration path;
its outdated sample download is tracked in
[#142](https://github.com/aganezov/ont_qc_mcp/issues/142).
