# YOLOv12n-WTMD research artifacts

Research materials for **YOLOv12n-WTMD: Geometry-guided prediction-scale reallocation for wind turbine blade damage detection**.

## Artifact status

This initial preparation contains machine-readable **manuscript-reported summaries**, dataset-source metadata, checkpoint identities, reported protocol specifications, and newly written assembly/validation utilities. The supplied inputs did not contain the historical detector/evaluator implementation, weights, split files, per-image predictions or fitted regressors. Their inclusion must be established by the local collection report. No network inference or independent reproduction of the paper's AP values has been performed by these utilities.

Consult `metadata/ARTIFACT_STATUS.json` and the generated assembly report for the current state. File presence, byte hashes, summary arithmetic and end-to-end metric reproduction are separate checks. A missing or changed file is reported explicitly; no synthetic observations or replacement model weights are supplied.

## Included files

| Path | Contents |
|---|---|
| `data/reported_results/` | 36 CSV files, 265 records, covering nine main tables and 24 appendix tables; original table cells and source locations are retained in JSON |
| `data/dataset_sources.json` | Exact upstream dataset versions and the reported study cohorts |
| `metadata/checkpoints.json` | Eight registered checkpoint digests and analysis roles; these are expected identities, not locally verified payloads in the initial preparation |
| `metadata/manuscript_sources.json` | Filenames and SHA-256 values of the supplied manuscript sources |
| `configs/environment_record.json` | Reported environment, explicitly distinguished from a complete dependency lock |
| `configs/reported_protocols.json` | Transcribed protocol settings, not reconstructed original run configurations |
| `tools/validate_reported_results.py` | New, standard-library checks of CSV transcription, identities and aggregate arithmetic |
| `recovered/` | Created only after original files are recovered; project-relative layout and source manifests preserve original bytes |

## Validate the reported summaries

Run with Python 3.8 or later; no installation is required for this step.

```bash
python tools/validate_reported_results.py
```

This verifies the summary tables and bookkeeping. It does not train a detector, load PyTorch checkpoints, rerun inference, refit CQR or recompute AP from individual predictions.

## Reproduce the study

Use `docs/REPRODUCIBILITY.md` to connect each reported endpoint to the required original implementation and inputs. The local version string Ultralytics 8.3.63 does not by itself identify the locally modified runtime or guarantee that the published checkpoints can be loaded. Recover the custom SF1SPPF module, parser changes and matching/evaluation implementations before attempting reproduction.

The primary complete-detector comparison is B-main/W-StageB. B-post has a separate evaluation role. Hierarchy, pooling, scale, CQR and sensitivity results retain their own declared protocols. Do not combine their AP values as additive improvements.

## Source datasets

Original images are obtained from the upstream Roboflow versions listed in `data/dataset_sources.json`. This preparation does not redistribute them or invent their licences. Study-specific image/annotation membership, the 19 excluded pairs, label conversion and historical run-list mappings must be supplied with the completed artifacts.

## Availability and citation

The draft in `docs/DATA_AVAILABILITY_DRAFT.md` must be completed only after the referenced files are actually available. No repository URL, DOI or publication DOI is invented. `CITATION.cff` identifies the associated research artifact preparation and supplied manuscript authors.

## Licensing

The AGPL-3.0 licence applies to newly authored repository utilities and any compatible project code released under that licence. Recovered third-party files retain their original notices and licences. Dataset images and pretrained model files are subject to their respective terms. See `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md`.
