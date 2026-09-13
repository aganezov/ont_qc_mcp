# API v2 contracts

`ont_qc_mcp.v2_contracts` freezes the request and response boundary for the proposed ten-tool API. It is a catalog skeleton only. Nothing imports it into `app_server`, so the server continues to advertise the existing 18 tools until the replacement is implemented and switched over in a later slice.

The numerical tools use the same top-level names for shared concepts: `path`, `reference_path`, `regions`, `selection`, `group_by`, `metrics`, `extra_args`, and `deadline_seconds`. Unknown fields and wrong native namespaces fail validation. Native extras are ordered argument arrays and default to empty arrays. Protected-option and typed-option conflict checks belong to V2-02; accepting an array here does not establish that its option combination is scientifically valid.

## Regions and grouping

Omitted `regions` means whole-file or whole-reference scope. An empty list is invalid. Explicit interval objects and BED use zero-based half-open coordinates. Samtools strings use one-based inclusive coordinates and must be normalized before execution. The retained GFF/GFF3 mode supports gene lookup only; this contract does not add GTF or a general annotation query language.

Normalized responses preserve request order, names, coordinates, and stable `region_1`, `region_2`, ... identifiers in `effective_request.normalized_regions`. Grouped read, alignment and variant results must also preserve these names. Effective metrics cannot repeat, and coverage grouping must follow the region/window combination. The effective request also states the resolved grouping, metrics, selection, native arguments, input path, and reference path.

For coverage grouped by region, rows have a one-to-one ordered correspondence with normalized intervals. Each row's ID, contig, coordinates, and name must match its requested interval. The union reference length is checked against the genomic union of normalized intervals across contigs. Window-row tiling and correspondence are backend acceptance work for V2-04 rather than an invariant claimed by this contract-only slice.

Read, alignment, and variant calls default to `combined`; `region` requires regions. Coverage resolves its natural row type from the request:

| Regions | Window size | Resolved rows |
| --- | --- | --- |
| omitted | omitted | contigs |
| supplied | omitted | requested intervals |
| omitted | supplied | windows within each reference contig |
| supplied | supplied | windows tiled separately within each requested interval |

The final interval window can be shorter. Coverage rows retain zero-depth bases. `union_summary` measures the union of overlapping requested intervals rather than the sum of overlapping row lengths.

## Numerical definitions and defaults

`read_qc` defaults to `length` and `read_quality`; distributions are opt-in named sections. For BAM/CRAM, regional overlap selects primary alignment records and conversion uses their complete stored sequences. A missing stored sequence is excluded from emitted sequences and counted in `conversion_exclusions`. If a selected primary record has absent QUAL, a request for `read_quality` or `quality_distribution` fails as a whole, so successful read-quality sections always report zero missing-quality reads. This initial limitation prevents samtools FASTQ conversion from turning missing quality into synthetic Q1 evidence. A length-only request can still use the stored sequence. Success waits for all stages; no partial nanoq result is returned after a late missing-QUAL or upstream failure.

V2 adapters normalize `mean_length` to `total_bases / read_count` rather than forwarding a rounded native display value; empty selections have zero total bases and null summaries. Known length percentiles are nonnegative and monotonic.

For combined read results, `selected_records = emitted_sequences + conversion_exclusions`, and any returned length or read-quality count equals `emitted_sequences`. Per-region rows can overlap and are deliberately not summed into these global unique-record counts.

`alignment_qc` defaults to `counts` and `mapping_quality`. Its default mask is 1796: unmapped, secondary, QC-failed, and duplicate records are excluded while supplementary records remain eligible. MAPQ 255 is missing, is retained only when `min_mapq=0`, and is excluded from the mean denominator. Optional `aligned_base_quality` uses stored qualities for M, `=`, or X query bases inside the interval and reports known and missing base denominators. An unmapped-inclusive whole-file request must explicitly remove bit `0x4` from `exclude_flags`; unmapped records are invalid for regional reports.

`include_unmapped=False` independently excludes unmapped records, even when a custom `exclude_flags` mask omits bit `0x4`. Adapters must apply both selection controls; for a samtools flag filter they must add bit `0x4` when this boolean is false. Unmapped eligibility requires both `include_unmapped=True` and an exclusion mask without bit `0x4`. The boolean does not override other filters.

The current identity parser copies cramino's aggregate `identity_stats.mean_identity`; it does not expose cramino's numerator, denominator, or number of records contributing to that mean. Cramino emits this value as a percentage (for example, `87.724...` in the pinned raw fixture), so v2 explicitly normalizes it to a fraction (`0.87724...`). Per-record identity availability counts remain null when the backend does not supply them. V2-05 must verify cramino's aggregation definition before claiming more than a backend-reported whole-selected-record mean.

The current samtools error-rate definition is concrete: `error rate` is the sum of NM tags divided by bases mapped according to CIGAR, and NM can include indels. If samtools separately reports `mismatches per base`, v2 returns that as `mismatch_rate` rather than relabeling it as the NM-derived rate. Explicit insertion/deletion rates, the COV reference-position count histogram, MPC one-based cycle/N/quality-stratified mismatch counts, and the insert-size histogram remain available. Samtools stats does not expose counts of records with and without NM in this aggregate, so those availability counts remain null. Identity and error-profile values describe whole selected records even when overlap with a region selected those records.

When alignment counts and mapping quality appear together, mapped plus unmapped records and known plus missing MAPQ records each equal the eligible record count. Secondary and supplementary counts are nonexclusive annotations of that population and cannot exceed its size. Available aggregate evidence must obey the effective unmapped, flag and MAPQ selection.

`coverage_qc` defaults to depth and breadth at thresholds 1, 10, and 20. It uses mosdepth-native selection controls only. Coordinates and `reference_bases` are structural row fields. When `depth` is requested, each row contains paired `depth_sum` and `mean_depth` fields; `mean_depth` is `depth_sum / reference_bases`. A zero-reference union has `depth_sum: 0` and `mean_depth: null`. When depth is not requested, both depth fields are omitted or null in rows and the union summary. When `breadth` is requested, every positive-length row contains exactly the requested thresholds in order, and each required finite fraction equals `bases_at_or_above / reference_bases`. When breadth is not requested, the breadth list is empty. Metric selection defines the response shape and does not promise reduced backend computation.

`variant_qc` defaults to general, SNP, and indel sections. Combined and per-region grouping are supported. Include and exclude expressions are mutually exclusive. Transition and transversion values are allele counts and need not sum to the SNP record count. They are either both available or both unavailable. V2 recomputes TS/TV from those counts; the ratio is null if the counts are unavailable or the transversion denominator is zero. This avoids treating a rounded native display value as independent evidence. Region normalization and bcftools execution are future implementation work.

## Supporting tools and errors

Length, quality, and insert-size distributions use strict v2 histogram bins with finite ordered bounds and nonnegative integer counts. Equal bounds remain valid for exact-value bins. Error-profile COV and MPC structures likewise use strict v2 count models rather than permissive legacy wire models.

`environment_status`, `header_info`, `bed_qc`, `run_summary`, `filter_reads`, and `igv_snapshots` retain their current core behavior. `run_summary` remains a path-only request and its response retains the parser's fixed, nonempty one-hour bins anchored at the first read start. The other request contracts keep existing meaningful controls, including typed chopper filters and IGV dynamic or prebuilt-batch modes with current display and rendering settings. Dynamic IGV regions retain strict zero-based half-open coordinates, names, and per-region `extra_commands`; top-level commands do not replace per-region commands. Existing response models are reused where their meaning already fits.

Validation errors contain field locations and codes. Execution errors identify the failed stage/backend and whether the request timed out or was cancelled. `partial_result_returned` is always false. Deadline propagation, draining, process-group cancellation, cleanup, backend algorithms, registration, and protected native-option routing are implemented in later slices.

Representative contract examples live in `tests/fixtures/api_v2/`. They are illustrative wire objects, not backend-computed scientific evidence. Issue #104 remains V2-03 work, and unknown-resource handling in #101 remains separate.
