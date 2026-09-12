# Regional alignment evidence

`regional_alignment_stats_tool` collects counts, mapping quality and basecall quality
for one or more regions in one indexed BAM/CRAM. It returns numerical evidence for
an agent or human to interpret. Comparisons between files remain with the caller.

## Request

```json
{
  "path": "/data/sample.bam",
  "regions": [
    {"chrom": "chr1", "start": 1000, "end": 1100, "name": "candidate"},
    {"chrom": "chr1", "start": 900, "end": 1000, "name": "left flank"}
  ],
  "exclude_flags": 1796,
  "min_mapq": 0
}
```

Coordinates are **zero-based and half-open**: start is included, end is excluded.
A one-base interval at the first reference base is `[0, 1)`. All intervals must be
nonempty and fit a contig declared in the alignment header. There is no implicit
clipping or unknown-contig-to-zero conversion. Requests accept 1–1024 intervals.
Use a one-element list for a single interval. Contig names are at most 1024
characters and optional names at most 256 characters.

The input must be local, coordinate-sorted, and have a usable matching index.
BAM index discovery checks `.bam.csi`, `.csi`, `.bam.bai`, then `.bai`; CRAM checks
`.cram.crai`, then `.crai`. The selected index is passed explicitly to samtools and
identified in the result. No index is generated or repaired. SAM input is not
supported by this indexed operation.

For CRAM, supply `reference_path`, pointing to a local uncompressed `.fa`,
`.fasta` or `.fna` beginning with `>`, with an existing `.fai`. All alignment-header
contig names and lengths must match that index. The child uses the supplied
reference with reference-cache and reference-search environment variables disabled.
CRAM decoding retains HTSlib's reference checksum checks. A supplied reference
for BAM undergoes the same name/length preflight, but does not establish that it
is the reference originally used for alignment. Embedded-reference CRAM still
requires the explicit reference under this initial contract.

## Selection and counting

Every metric describes **retained alignment records**. The default exclusion mask,
1796, excludes secondary (`256`), QC-failed (`512`) and duplicate (`1024`) records,
as well as unmapped (`4`). Supplementary alignments remain included. Mapped-only
eligibility always applies, even if the caller removes bit 4 from the mask.
`exclude_flags` accepts 0–65535. All read groups are included; header read-group
IDs and sample names are returned for context.

`min_mapq` accepts 0–254. MAPQ 255 means unavailable: those records are retained
when the minimum is zero, and excluded when it is positive. Their values never
enter a mapping-quality mean. The response returns the effective policy. The
exclusion mask (with mapped-only bit 4 added) and minimum MAPQ are passed to
samtools as `-F` and `-q`. The accumulator also applies the MAPQ255 missing-value
rule, which a numerical samtools threshold alone cannot express.

Paired, supplementary or repeated records can represent the same biological read
or base more than once. The tool does not deduplicate by read name. Flag-category
counts are nonexclusive and describe retained records, so they do not count the
records excluded by the policy.

Each requested interval gets its original coordinates/name and a stable positional
`region_id` such as `region_1`. Overlapping and duplicate intervals remain separate
and preserve request order. A stored alignment is retrieved once by the batch
iterator and contributes to each requested interval it overlaps. Do not sum
region results as a unique-read or unique-base total.

## Measurements

| Field | Definition |
|---|---|
| `span_overlapping_alignments` | Retained records with reference-span overlap. A deletion or reference skip can produce overlap without aligned query bases. |
| `aligned_base_alignments` | Retained records contributing at least one aligned query base inside the interval. |
| `aligned_query_bases` | Query bases aligned by CIGAR `M`, `=` or `X` inside the interval. |
| `mapq_known_alignments`, `mapq_missing_alignments` | Counts of available MAPQ and MAPQ255 among span-overlapping records. Their sum equals the span count. |
| `mean_mapq` | Arithmetic mean of available MAPQ over span-overlapping records. Null when none are available. |
| `quality_known_bases`, `quality_missing_bases` | Counts of available and unavailable QUAL among aligned query bases. Their sum equals `aligned_query_bases`. |
| `mean_base_quality` | Arithmetic mean Phred of available stored base qualities for aligned query bases inside the interval. Null when none are available. |
| `secondary_alignments`, `supplementary_alignments`, `duplicate_alignments`, `qcfail_alignments`, `reverse_alignments` | Nonexclusive flag counts among retained span-overlapping records. |

Insertions and soft clipping advance the query offset but contribute no regional
bases. Deletions and reference skips advance the reference offset but contribute
no query bases or qualities. Hard clipping and padding advance neither. Reverse
alignments use the sequence/quality orientation already stored in SAM; qualities
are not reversed a second time. There is no minimum base-quality filter.

For example, SAM POS101 with CIGAR `2S3M2I2M2D2M3N1M1S` and stored qualities
10 through 22 contributes qualities 14, 17, 18 and 19 to `[102,108)`. The regional
mean is **17**, whereas its whole-read arithmetic mean is 16. In `[105,107)`,
the same record gives one span overlap, zero contributing alignments and no
base-quality mean.

These means describe reported confidence. They are not empirical accuracy,
alignment identity, Phred of the mean error probability, or a variant-support
classification. A zero count differs from an unavailable mean. A failure returns
an error; it never returns accumulated partial measurements as complete.

## Resources, provenance and limits

The operation reads the tool version and alignment header, then streams one
indexed samtools multi-region pass. It does not materialize a regional BAM/FASTQ
or retain SAM output. Optional tags are removed from the SAM stream because these
measurements do not use them. A small private BED is removed after success or
failure. Stdout lines are limited to 16 MiB and retained stderr to 64 KiB per
command; the header/FASTA index limit is 16 MiB. Buffer limits reject oversized
records rather than truncate measurements. Memory also depends on the current
record's parsed CIGAR and the interval index, and HTSlib has its own buffers.

`MCP_TIMEOUT_SAMTOOLS` is an overall request budget across subprocess stages;
`MCP_THREADS_SAMTOOLS` controls samtools additional threads. Cancellation and errors
stop owned processes and clean temporary files. This is a foreground operation,
not a persistent job. Runtime depends on indexed blocks, local depth, read lengths,
interval count and storage. No WGS runtime or memory bound has been measured.

The response contains input/index/reference paths, sizes, modification times,
device/inode identities, selected alignment-header references, tool/server versions,
effective settings, definitions, warnings and `complete: true`. File metadata is
checked again after analysis; a detected change fails the call. These are metadata
identities, not content checksums or proof that an index matches its alignment.
Header MD5 values are declarations; BAM reference identity is not verified.

Two explicit format limitations prevent silent incorrect results:

- Contig names beginning with `#` are rejected because BED interprets them as
  comments. Other legal names such as `track` remain supported.
- In a one-base record, samtools emits both native BAM Q9 and unavailable quality
  as SAM QUAL `*`. If such a retained record contributes an aligned base to a
  requested interval, the tool returns an ambiguity error. This also rejects a
  genuinely missing quality on that one-base record. Multi-base missing qualities
  remain supported; filtered records and span-only overlaps do not trigger this
  limitation.

See the primary [samtools view documentation](https://www.htslib.org/doc/samtools-view.html),
[SAM specification](https://samtools.github.io/hts-specs/SAMv1.pdf), and
[HTSlib reference handling](https://www.htslib.org/doc/reference_seqs.html).
