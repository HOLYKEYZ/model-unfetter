# Public Data Release

This repository includes three public data areas:

- `pipeline_collected_data/`: public programming Q&A and discussion pairs collected from Reddit, Hacker News, and Stack Exchange.
- `collected_training_data/`: GitHub source/documentation snippets from permissively licensed repositories, with per-row license identifiers.
- `external_datasets/`: third-party datasets mirrored locally for reproducibility.

## Licensing

There is no single license grant for all dataset rows. Each row remains under the terms of its original source.

- GitHub rows keep their upstream repository licenses, recorded in each row and summarized in `collected_training_data/collector_report.json`.
- Stack Exchange rows are user contributions under Creative Commons Attribution-ShareAlike terms.
- Hacker News rows come from public Hacker News API / Algolia-indexed public discussion data.
- Reddit rows are public Reddit posts/comments; review Reddit's current API and content terms before commercial redistribution or model training.
- External Hugging Face datasets keep their upstream dataset license and attribution.

Do not treat this release as relicensing upstream content under this repository's code license.

## Exclusions

The release intentionally excludes local cache folders, downloaded verification folders, private sessions, generated private curated datasets, model checkpoints, and Kaggle run artifacts.
