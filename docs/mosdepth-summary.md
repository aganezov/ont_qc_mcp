# Mosdepth coverage summaries

`coverage_stats_tool` and the coverage section of `alignment_summary_tool` parse
mosdepth's `.summary.txt` file. They return contig-level coverage. Supplying a
window makes mosdepth produce additional region output, but these tools still
return the contig summaries rather than individual windows.

| Field | Meaning |
| --- | --- |
| `coverage_by_contig` | Name, length, and mean depth for each contig reported in the summary. Whole-genome and region aggregate rows are excluded. |
| `mean_depth` | Length-weighted mean of the reported contig means. The calculation uses mosdepth's printed means, which are rounded. |
| `mean_depth_unweighted` | Arithmetic mean of the reported contig means, giving each contig equal weight. |
| `low_coverage_regions` | Whole-contig intervals whose mean depth is strictly below `low_cov_threshold`. These are not local coverage gaps or failing windows. |
| `median_depth`, per-contig `median_depth` | Unavailable and returned as null. |
| `coverage_distribution` | Empty because this parser does not consume the distribution files. |

For example, a 10 kb contig at 10x and a 30 kb contig at 1x give a length-weighted
mean of 3.25x and an unweighted mean of 5.5x. With threshold 5, only the second
contig is reported as low coverage, spanning `[0, 30000)`. The aggregate `total`
row is not a genomic location.

Mosdepth emits its whole-genome totals at the end of the summary. With `--by`,
each contig row is followed by a corresponding `_region` summary, and a
`total_region` row follows the final `total` row. The parser uses that structure,
so real contigs named `total`, `total_region`, or ending in `_region` remain
valid. It also distinguishes the actual header from contigs such as `chrom`
and `chromosome1`. See the [mosdepth 0.3.14 implementation](https://github.com/brentp/mosdepth/blob/v0.3.14/mosdepth.nim).

The parser does not add contigs absent from the summary. For explicit BED,
location, or gene intervals and coverage threshold percentages, use
`targeted_coverage_tool`, which reads separate region and threshold files.
