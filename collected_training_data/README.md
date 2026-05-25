# Collected Training Data

This folder contains public GitHub source/documentation snippets collected for coding-language-model experiments.

## Files

- `github_code.jsonl`: 1,505 rows from public GitHub repositories.
- `collector_report.json`: repository-level source summary and row counts.

## Schema

Each `github_code.jsonl` row has:

```json
{"source": "github", "repo": "owner/name", "license": "spdx-or-short-id", "path": "repo/path", "text": "file content"}
```

## Licensing

Rows are not relicensed by this repository. Each row keeps the upstream repository license recorded in its `license` field.

The current collection includes permissive licenses such as MIT, BSD-3-Clause, and Apache-2.0. Preserve source repository attribution, path, and license fields when reusing this data.
