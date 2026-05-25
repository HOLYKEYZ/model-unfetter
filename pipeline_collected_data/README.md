# Pipeline Collected Data

This folder contains public programming Q&A/discussion pairs collected by source-specific collectors.

## Files

- `reddit/reddit_coding.jsonl`: 80 Reddit public post/comment pairs.
- `reddit/collector_report.json`: Reddit collection summary.
- `hackernews/hackernews_coding.jsonl`: 403 Hacker News public discussion pairs.
- `hackernews/collector_report.json`: Hacker News collection summary.
- `stackexchange/stackexchange_coding.jsonl`: 205 Stack Exchange Q&A pairs.
- `stackexchange/report.json`: Stack Exchange collection summary.

## Schemas

Reddit and Hacker News rows use:

```json
{"source": "string", "url": "string", "prompt": "string", "completion": "string", "metadata": {}}
```

Stack Exchange rows use:

```json
{"source": "stackexchange", "prompt": "string", "completion": "string", "messages": [], "metadata": {}, "id": "string"}
```

## Licensing And Attribution

This folder is a mixed-source public data collection. It is not relicensed under the repository's code license.

- Stack Exchange user contributions are Creative Commons Attribution-ShareAlike content. Keep the `metadata.link`, `question_id`, and `answer_id` fields when redistributing.
- Hacker News rows are public discussion data collected from public APIs/indexes. Keep `url`, story/comment ids, and source metadata.
- Reddit rows are public posts/comments. Keep `url`, permalink, post/comment ids, subreddit, and source metadata. Review Reddit's current terms before commercial redistribution or model training.
