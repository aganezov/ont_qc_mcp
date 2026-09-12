# Samtools statistics fields

`alignment_error_profile_tool` parses the text emitted by `samtools stats`. Field definitions follow [samtools 1.24](https://www.htslib.org/doc/1.24/samtools-stats.html).

| Field | Meaning |
| --- | --- |
| `mismatch_rate` | Retained compatibility name. The usual source is the SN `error rate`: the sum of NM tags divided by bases mapped according to CIGAR. NM can include indels, so this is not a substitution-only rate. A separately reported `mismatches per base` value takes precedence. |
| `insertion_rate`, `deletion_rate` | Populated only from explicit corresponding per-base rate fields. Samtools 1.24 does not normally emit those fields, so they remain null. |
| `coverage_histogram` | Reference-position counts by inclusive integer depth range. Each record has `start`, `end`, and `count`; `end: null` means no upper bound. These are the sites counted by samtools, not a whole-reference histogram that enumerates uncovered positions. |
| `mismatch_counts_by_cycle` | MPC records containing an explicit one-based `cycle`, an `n_count`, and `mismatches_by_quality`. List index 0 is Q0, index 1 is Q1, and so on. Quality counts exclude N bases. No per-cycle denominator is provided, so these are counts, not rates. |
| `mismatch_by_cycle` | Deprecated rate field, left null by the parser. Use `mismatch_counts_by_cycle` for MPC counts. |
| `gc_coverage` | Retired for ONT QC; always null when parsed. Samtools GCD estimates are not direct mapped-depth measurements. |

Missing sections remain null. Present zero-valued rows are retained. Coverage underflow `[<25]` is represented as `start: 0, end: 24`; this mathematical lower bound does not imply that samtools counted uncovered reference positions. Overflow `[19<]` is `start: 20, end: null`. The parser follows the printed range, which can differ from the requested coverage maximum when the step is greater than one.

## Example

Twenty 50-base reads aligned at the same reference position give 50 sites at depth 20. If their NM tags sum to 23, samtools reports:

```text
SN	error rate:	2.300000e-02	# mismatches / bases mapped (cigar)
COV	[20-20]	20	50
```

The corresponding output excerpt is:

```json
{
  "mismatch_rate": 0.023,
  "coverage_histogram": [{"start": 20, "end": 20, "count": 50}],
  "mismatch_by_cycle": null,
  "mismatch_counts_by_cycle": null
}
```

Samtools emits MPC only when given a reference with `-r`. The current MCP wrapper does not expose that option, so the cycle fields normally remain null. Direct parsing of reference-backed samtools output preserves the MPC counts. For example, `MPC` fields `3, 2, 4, 0, 1` describe cycle 3, two N bases, four Q0 mismatches, no Q1 mismatches, and one Q2 mismatch:

```json
{
  "cycle": 3,
  "n_count": 2,
  "mismatches_by_quality": [4, 0, 1]
}
```

## Why GC-depth output is retired

The old parser treated a GCD depth percentile as an integer histogram count. Keeping the fields correctly would still leave a measurement problem for ONT: samtools estimates GCD depth from bin alignment counts multiplied by the global mean read length, divided by bin size. Its implementation assumes reads are short relative to the default 20 kb bins. That global mean also includes unmapped reads. [Pinned assumptions](https://github.com/samtools/samtools/blob/dc71c7274044d1050ccb64901731373ec7e915b6/stats.c#L26-L35), [mean length and scaling](https://github.com/samtools/samtools/blob/dc71c7274044d1050ccb64901731373ec7e915b6/stats.c#L1621-L1627), [GCD output](https://github.com/samtools/samtools/blob/dc71c7274044d1050ccb64901731373ec7e915b6/stats.c#L1895-L1927).

A controlled samtools 1.24 example keeps thirty mapped 1 kb reads unchanged and adds ten unmapped 20 kb reads. Mapped CIGAR bases remain 30,000 and the COV histogram remains 3,000 sites at depth 10, but the GC50 depth estimates change from 0.500 to 2.875. The estimates therefore respond to read-length composition even when mapped coverage is identical. This does not establish how often that effect occurs in real ONT datasets.

For this ONT-focused server, `gc_coverage` remains present and null for compatibility. SN rates, COV coverage, MPC counts and insert-size parsing are unchanged by this retirement. The JSON files in [the older examples](tool-output-examples.md) remain historical snapshots and should be regenerated before use with the current API.
