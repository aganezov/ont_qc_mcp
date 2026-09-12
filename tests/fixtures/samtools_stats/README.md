# samtools stats 1.24 fixtures

Run `python tests/fixtures/samtools_stats/generate.py --samtools samtools` with
samtools 1.24. Use `--output-dir DIRECTORY` to regenerate elsewhere. The generator
checks the version and retains literal SN, COV, and MPC rows, including tab-separated
comments. `provenance.json` records commands, the upstream commit, and file hashes.
No generated command headers or machine paths are retained.

The reference contains 100 A bases. Twenty 50 bp reads cover reference positions
1–50, giving 50 positions at depth 20. Every read has C at cycle 2; ten have Q0
there and ten have Q40. Three reads have N at cycle 1. Their NM values are 2; the
other seventeen NM values are 1. Thus the SN NM-derived error rate is 23/1000 =
0.023. Reference-assisted MPC reports N count 3 at cycle 1, zero quality-stratified
mismatches there, and ten Q0 plus ten Q40 mismatches at cycle 2. Samtools also prints an all-zero cycle 51 for these
50 bp reads; the fixture retains that row.

Coverage options exercise default `[20-20]`, overflow `[19<]` with `-c 5,15,5`, and
underflow `[<25]` with `-c 25,35,5`, and the multi-depth `[20-24]` interval
with `-c 15,25,5`. These are samtools-covered positions; the fixture
does not infer or enumerate uncovered reference positions. Synthetic parser controls
for other ranges, sparse/reordered cycles, zero records, and malformed fields live
in `tests/test_samtools_stats.py`.
