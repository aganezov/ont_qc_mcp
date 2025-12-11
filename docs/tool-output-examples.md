# MCP Tool Output Examples (real fixtures)

- Generated on 2025-12-11 with `scripts/with-env.sh python` (see regeneration snippet below) against the real fixtures in `tests/fixtures/real/` plus a tiny synthetic high-depth BAM.
- Full raw outputs live in `docs/tool_output_examples.json`; a trimmed digest lives in `docs/tool_output_examples_summary.json`.
- Inputs:
  - FASTQ: `tests/fixtures/real/haplotag.large.fq.gz`
  - BAM: `tests/fixtures/real/haplotag.large.bam`
  - VCF: `tests/fixtures/real/haplotag.large.vcf.gz`
  - High-depth BAM: `tests/fixtures/synthetic/highdepth.bam` (50 x 100bp reads over 1kb)
- Notes: nanoq v0.10.0 JSON omits histogram blocks, so histograms stay empty though length/quality stats are present. Cramino’s current JSON lacks mapped/unmapped counts, so those remain zero even though identity metrics are populated. The real BAM is extremely downsampled; coverage is near-zero there, but the synthetic BAM shows non-zero coverage.

## Quick regeneration
Run this to refresh both JSON files with the real + synthetic fixtures:
```
scripts/with-env.sh python - <<'PY'
from pathlib import Path
import anyio, json, tempfile
from ont_qc_mcp.tools import (
    env_check, qc_reads, filter_reads, read_length_distribution, qscore_distribution,
    read_length_distribution_bam, qscore_distribution_bam, qc_alignment, coverage_stats,
    alignment_error_profile, alignment_summary, header_metadata_lookup, serialize_model,
)
from ont_qc_mcp.config import ToolPaths

FASTQ = Path('tests/fixtures/real/haplotag.large.fq.gz')
BAM = Path('tests/fixtures/real/haplotag.large.bam')
VCF = Path('tests/fixtures/real/haplotag.large.vcf.gz')
HD_BAM = Path('tests/fixtures/synthetic/highdepth.bam')
DOCS_DIR = Path('docs')
DOCS_DIR.mkdir(exist_ok=True)

tools = ToolPaths()

async def main():
    outputs = {}
    outputs['env_status'] = serialize_model(env_check(tools))
    outputs['qc_reads_fastq_tool'] = serialize_model(qc_reads(str(FASTQ), tools=tools))

    with tempfile.NamedTemporaryFile(suffix='.fastq', delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        outputs['filter_reads_fastq_tool'] = serialize_model(
            filter_reads(str(FASTQ), tools=tools, output_fastq=str(out_path))
        )
    finally:
        out_path.unlink(missing_ok=True)

    outputs['read_length_distribution_fastq_tool'] = serialize_model(
        read_length_distribution(str(FASTQ), tools=tools)
    )
    outputs['qscore_distribution_fastq_tool'] = serialize_model(
        qscore_distribution(str(FASTQ), tools=tools)
    )

    outputs['read_length_distribution_bam_tool'] = serialize_model(
        await read_length_distribution_bam(str(BAM), tools=tools)
    )
    outputs['qscore_distribution_bam_tool'] = serialize_model(
        await qscore_distribution_bam(str(BAM), tools=tools)
    )

    outputs['qc_alignment_tool'] = serialize_model(qc_alignment(str(BAM), tools=tools))
    outputs['coverage_stats_tool'] = serialize_model(coverage_stats(str(BAM), tools=tools))
    outputs['alignment_error_profile_tool'] = serialize_model(
        alignment_error_profile(str(BAM), tools=tools)
    )
    outputs['alignment_summary_tool'] = serialize_model(alignment_summary(str(BAM), tools=tools))

    outputs['alignment_summary_tool_highdepth'] = serialize_model(
        alignment_summary(str(HD_BAM), tools=tools)
    )

    bam_meta = header_metadata_lookup(str(BAM), file_type='bam', tools=tools)
    outputs['header_metadata_tool_bam'] = {
        'summary': bam_meta.summary,
        'payload': serialize_model(bam_meta),
    }
    vcf_meta = header_metadata_lookup(str(VCF), file_type='vcf', tools=tools)
    outputs['header_metadata_tool_vcf'] = {
        'summary': vcf_meta.summary,
        'payload': serialize_model(vcf_meta),
    }

    out_full = DOCS_DIR / 'tool_output_examples.json'
    out_full.write_text(json.dumps(outputs, indent=2))

    def head(hist, n=5):
        return hist[:n] if hist else []

    summary = {
        'env_status': outputs['env_status'],
        'qc_reads_fastq_tool': {
            k: outputs['qc_reads_fastq_tool'].get(k)
            for k in ['read_count', 'total_bases', 'min_len', 'max_len', 'mean_len', 'median_len', 'n50', 'mean_qscore', 'median_qscore', 'gc_content']
        } | {
            'length_histogram_head': head(outputs['qc_reads_fastq_tool'].get('length_histogram')),
            'qscore_histogram_head': head(outputs['qc_reads_fastq_tool'].get('qscore_histogram')),
        },
        'filter_reads_fastq_tool': outputs['filter_reads_fastq_tool'],
        'read_length_distribution_fastq_tool': {
            'percentiles': outputs['read_length_distribution_fastq_tool'].get('percentiles'),
            'histogram_head': head(outputs['read_length_distribution_fastq_tool'].get('histogram')),
        },
        'qscore_distribution_fastq_tool': {
            'mean_qscore': outputs['qscore_distribution_fastq_tool'].get('mean_qscore'),
            'median_qscore': outputs['qscore_distribution_fastq_tool'].get('median_qscore'),
            'histogram_head': head(outputs['qscore_distribution_fastq_tool'].get('histogram')),
        },
        'read_length_distribution_bam_tool': {
            'percentiles': outputs['read_length_distribution_bam_tool'].get('percentiles'),
            'histogram_head': head(outputs['read_length_distribution_bam_tool'].get('histogram')),
        },
        'qscore_distribution_bam_tool': {
            'mean_qscore': outputs['qscore_distribution_bam_tool'].get('mean_qscore'),
            'median_qscore': outputs['qscore_distribution_bam_tool'].get('median_qscore'),
            'histogram_head': head(outputs['qscore_distribution_bam_tool'].get('histogram')),
        },
        'qc_alignment_tool': {
            k: outputs['qc_alignment_tool'].get(k)
            for k in ['total_reads', 'mapped', 'unmapped', 'mean_length', 'median_length', 'n50', 'mean_identity', 'median_identity']
        } | {
            'length_histogram_head': head(outputs['qc_alignment_tool'].get('length_histogram')),
            'mapq_histogram_head': head(outputs['qc_alignment_tool'].get('mapq_histogram')),
        },
        'coverage_stats_tool': {
            'mean_depth': outputs['coverage_stats_tool'].get('mean_depth'),
            'mean_depth_unweighted': outputs['coverage_stats_tool'].get('mean_depth_unweighted'),
            'coverage_distribution_head': head(outputs['coverage_stats_tool'].get('coverage_distribution')),
            'coverage_by_contig_head': outputs['coverage_stats_tool'].get('coverage_by_contig', [])[:3],
        },
        'alignment_error_profile_tool': {
            k: outputs['alignment_error_profile_tool'].get(k)
            for k in ['mismatch_rate', 'insertion_rate', 'deletion_rate']
        } | {
            'coverage_histogram_head': head(outputs['alignment_error_profile_tool'].get('coverage_histogram')),
            'gc_coverage_head': head(outputs['alignment_error_profile_tool'].get('gc_coverage')),
        },
        'alignment_summary_tool': {
            'alignment_keys': list(outputs['alignment_summary_tool'].keys()),
            'alignment_mapped': outputs['alignment_summary_tool'].get('alignment', {}).get('mapped') if isinstance(outputs['alignment_summary_tool'].get('alignment'), dict) else None,
            'coverage_mean_depth': outputs['alignment_summary_tool'].get('coverage', {}).get('mean_depth') if isinstance(outputs['alignment_summary_tool'].get('coverage'), dict) else None,
        },
        'alignment_summary_tool_highdepth': {
            'alignment_keys': list(outputs['alignment_summary_tool_highdepth'].keys()),
            'alignment_mapped': outputs['alignment_summary_tool_highdepth'].get('alignment', {}).get('mapped') if isinstance(outputs['alignment_summary_tool_highdepth'].get('alignment'), dict) else None,
            'coverage_mean_depth': outputs['alignment_summary_tool_highdepth'].get('coverage', {}).get('mean_depth') if isinstance(outputs['alignment_summary_tool_highdepth'].get('coverage'), dict) else None,
        },
        'header_metadata_tool_bam': {
            'summary': outputs['header_metadata_tool_bam']['summary'],
            'references': len(outputs['header_metadata_tool_bam']['payload'].get('references', [])),
            'samples': [s.get('name') for s in outputs['header_metadata_tool_bam']['payload'].get('samples', [])],
        },
        'header_metadata_tool_vcf': {
            'summary': outputs['header_metadata_tool_vcf']['summary'],
            'references': len(outputs['header_metadata_tool_vcf']['payload'].get('references', [])),
            'samples': [s.get('name') for s in outputs['header_metadata_tool_vcf']['payload'].get('samples', [])],
        },
    }

    out_summary = DOCS_DIR / 'tool_output_examples_summary.json'
    out_summary.write_text(json.dumps(summary, indent=2))
    print('Wrote', out_full, 'and', out_summary)

anyio.run(main)
PY
```

## Tool-by-tool examples
Values come from `docs/tool_output_examples_summary.json` unless noted; see the full JSON for complete payloads.

- **env_status** — call `env_status` (no args). Available: nanoq, chopper, cramino, mosdepth, samtools; resolved paths point to `/Users/saganezov/.cargo/bin/nanoq` and `/Users/saganezov/miniforge3/envs/ont-qc-mcp/bin/*`.

- **qc_reads_fastq_tool** — call with `{"path": "tests/fixtures/real/haplotag.large.fq.gz"}`.
  - Output excerpt: `read_count=221`, `total_bases=1,268,233`, `mean_len=5738`, `median_len=5087`, `n50=7476`, `mean_qscore≈33`; histograms remain empty in this nanoq build.

- **filter_reads_fastq_tool** — call with `{"path": ".../haplotag.large.fq.gz", "output_fastq": "/tmp/out.fastq"}`.
  - Output excerpt:
    ```json
    {
      "command": ["chopper", "--input", "tests/fixtures/real/haplotag.large.fq.gz", "--threads", "4"],
      "input_reads": null,
      "output_reads": null,
      "filtered_reads": null,
      "output_fastq": "/tmp/out.fastq"
    }
    ```
    Counts are null because this chopper build does not emit JSON stats.

- **read_length_distribution_fastq_tool** — call with the FASTQ path.
  - Output excerpt: percentiles all `null`, histogram empty (nanoq JSON lacks the histogram block).

- **qscore_distribution_fastq_tool** — call with the FASTQ path.
  - Output excerpt: `mean_qscore≈33`, `median_qscore≈33`, histogram empty.

- **read_length_distribution_bam_tool** — call with `{"path": "tests/fixtures/real/haplotag.large.bam"}`.
  - Output excerpt: percentiles all `null`, histogram empty (same nanoq JSON limitation when streaming via samtools).

- **qscore_distribution_bam_tool** — call with the BAM path.
  - Output excerpt: qscore stats `≈33`, histogram empty.

- **qc_alignment_tool** — call with the BAM path.
  - Output excerpt:
    ```json
    {
      "total_reads": 221,
      "mapped": 0,
      "length_histogram": [
        {"start": 0.0, "end": 2000.0, "count": 40},
        {"start": 2000.0, "end": 4000.0, "count": 65},
        {"start": 4000.0, "end": 6000.0, "count": 42},
        {"start": 6000.0, "end": 8000.0, "count": 33},
        {"start": 8000.0, "end": 10000.0, "count": 17}
      ],
      "mean_identity": 87.7
    }
    ```
    The length histogram comes from cramino; mapped/unmapped stay zero because the current cramino JSON lacks those counts even though identity metrics are present.

- **coverage_stats_tool** — call with the BAM path.
  - Output excerpt: `mean_depth=0.0`, `coverage_distribution=[]`, first contigs show `mean_depth=0.0` (the tiny BAM over chromosome-length references rounds to ~0 depth).

- **alignment_error_profile_tool** — call with the BAM path.
  - Output excerpt: mismatch/indel rates `null`; `gc_coverage` has three bins (0.0, 22.1, 48.0); coverage histogram empty. This reflects the limited `samtools stats` signal on the tiny BAM.

- **alignment_summary_tool** — call with the BAM path.
  - Output excerpt: combines the above; `alignment_mapped=0`, `coverage_mean_depth=0.0`, errors `null`.

- **alignment_summary_tool (high-depth BAM)** — call with `{"path": "tests/fixtures/synthetic/highdepth.bam"}`.
  - Output excerpt: `coverage_mean_depth=5.0` with non-zero coverage over `chrHD`; alignment section mirrors the synthetic 50 reads.

- **header_metadata_tool (BAM)** — call with `{"path": ".../haplotag.large.bam", "file_type": "bam"}`.
  - Summary text: `BAM header; 24 contigs (e.g., chr10, chr11, chr12, …); samples: NA12878; programs: minimap2, minimap2; sort order=coordinate`.

- **header_metadata_tool (VCF)** — call with `{"path": ".../haplotag.large.vcf.gz", "file_type": "vcf"}`.
  - Summary text: `VCF header; 3 contigs (e.g., chr1, chr2, chr3); samples: NA12878; 1 FORMAT fields`.

## Notes and gaps
- nanoq v0.10.0 JSON omits histogram blocks, so histograms/percentiles are empty even though length/quality stats are present.
- cramino’s current JSON lacks mapped/unmapped counts; identities come through, counts stay zero.
- The real BAM is extremely downsampled relative to whole-genome contig lengths (coverage ~0); the synthetic high-depth BAM is included to show non-zero coverage paths.
- Flag schemas, recipes, and guidance resources remain available via `tool://flags/{tool}`, `tool://recipes/{tool}`, and `tool://guidance/{tool}`.
