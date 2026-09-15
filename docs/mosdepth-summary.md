# Coverage QC measurements

`coverage_qc` reports reference-domain depth and breadth from an indexed BAM or
CRAM. It returns contig rows by default, requested interval rows when `regions`
is supplied, and window rows when `window_size` is supplied. Coordinates are
zero-based and half-open.

| Field | Meaning |
| --- | --- |
| `reference_bases` | Exact number of reference positions in the row. |
| `depth_sum` | Integer sum of depth over the row, present when `depth` is requested. |
| `mean_depth` | `depth_sum / reference_bases`, without deriving the sum from rounded mosdepth means. |
| `median_depth` | Pinned mosdepth lower-middle depth, present only for `depth_statistic: "median"`. Zero-depth bases are included. |
| `breadth[].bases_at_or_above` | Native integer count of bases meeting a requested threshold. |
| `breadth[].fraction_at_or_above` | Count divided by `reference_bases`, in the range 0 to 1. |
| `union_summary` | Depth over the genomic union of requested intervals, so overlapping or repeated intervals are not double-counted. |

Region rows retain request order, repeated coordinates, names, and stable IDs.
The union reference length is checked against the normalized genomic union.
Whole-reference windows use IDs such as `chrom.window_N`; interval windows use
`region_N.window_M` and tile independently from each requested interval's start.

Depth requests read mosdepth's temporary integer run-length per-base output and
require complete, nonoverlapping evidence for every represented contig. Pinned
mosdepth can omit wholly zero-depth contigs, so an internal 1X threshold count
must corroborate zero covered bases before the adapter supplies an all-zero row.
Partial contigs, negative evidence, truncated summaries, and disagreements across
native outputs fail the request. Breadth-only calls suppress per-base output.

Caller-selected filters live under `selection`; additional safe native mosdepth
arguments live under `extra_args.mosdepth`. Output-routing, region, threshold,
median, and parser-owned switches are protected. Temporary BED and mosdepth files
are removed after success, failure, timeout, or cancellation.

For alignment-record counts, MAPQ, identity, or error evidence, call
`alignment_qc`. The public recipe at `tool://recipes/alignment_qc` shows the two
calls needed when alignment and coverage reports must be collected together.
