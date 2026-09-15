# Regional alignment QC

`alignment_qc` reports alignment-record and aligned-base evidence for a whole
BAM/CRAM, the genomic union of requested intervals, or each requested interval in
order. It does not report reference coverage; use `coverage_qc` for that domain.

```json
{
  "path": "/data/sample.bam",
  "regions": [
    {"chrom": "chr1", "start": 1000, "end": 1100, "name": "candidate"},
    {"chrom": "chr1", "start": 900, "end": 1000, "name": "left flank"}
  ],
  "group_by": "region",
  "metrics": ["counts", "mapping_quality", "aligned_base_quality"],
  "selection": {"exclude_flags": 1796, "min_mapq": 0}
}
```

List and BED coordinates are zero-based and half-open. Samtools region strings
are one-based and inclusive; GFF3 gene coordinates are normalized to the same
half-open representation. All intervals must be nonempty and within declared
contig bounds. Requests accept at most 1,024 intervals and preserve repeated or
overlapping entries with stable `region_N` IDs.

Regional BAM/CRAM requests require an existing matching index. CRAM also requires
an explicit indexed uncompressed FASTA. The server does not generate or repair
indexes. It records path, index, and reference identities and rejects detected
changes during a call; these metadata checks are not content checksums or proof
that a BAM used a particular reference.

## Population and measurements

Every requested metric describes the same selected alignment-record population.
The default exclusion mask, 1796, excludes unmapped, secondary, QC-failed, and
duplicate records while retaining supplementary records. `include_unmapped` is
available only for whole-file reporting and requires an exclusion mask without
bit `0x4`. MAPQ 255 is unavailable rather than numeric zero. A positive
`min_mapq` excludes unavailable MAPQ records.

| Section | Meaning |
| --- | --- |
| `counts` | Eligible, mapped, unmapped, secondary, and supplementary record counts. |
| `mapping_quality` | Known and missing MAPQ denominators plus the mean over known values. |
| `aligned_base_quality` | M/=/X query bases inside the selected domain, known and missing quality denominators, and the mean over known bases. |
| `identity` | Cramino whole-selected-record mean identity normalized to a fraction. |
| `error_profile` | NM error rate, explicit mismatch/indel rates when available, and retained samtools count structures. |

Combined regional requests select the genomic union once, so a record overlapping
two intervals is counted once. Region grouping preserves separate rows and the
same record can contribute to more than one row. Paired, supplementary, or
repeated records are not deduplicated by read name.

Aligned-base quality counts only query bases consumed by CIGAR `M`, `=`, or `X`
inside the selected interval. Insertions and clipping do not contribute regional
reference bases; deletions and skips can create span overlap without aligned query
bases. Missing QUAL and MAPQ retain explicit denominators. A one-base SAM QUAL
`*` is ambiguous between missing quality and native Q9, so a contributing record
with that representation fails an aligned-base-quality request.

Only backends required by `metrics` run. All stages share one optional request
deadline. Timeouts, cancellation, and backend failures return no partial result,
stop owned processes, and remove temporary regional files. Additional safe native
arguments are accepted through `extra_args.samtools_view`,
`extra_args.samtools_stats`, and `extra_args.cramino`; population, reference,
output, and parser-owned options are protected.
