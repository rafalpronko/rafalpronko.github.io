"""Evidence labels: the sentences a reader needs to answer each query.

Labels point at (doc_id, sentence_index), never at chunks, so one label set serves
both chunking policies. Only metrics/experiment code reads this module.
"""

EVIDENCE: dict[str, frozenset[tuple[str, int]]] = {
    # 'It accepts at most 500 rows' is useless without the sentence naming the endpoint.
    "q01": frozenset({("export-api", 0), ("export-api", 1)}),
    "q02": frozenset({("search-api", 0), ("search-api", 1)}),
    "q03": frozenset({("errors", 2)}),
    "q04": frozenset({("billing", 0)}),
    "q05": frozenset({("billing", 1)}),
    "q06": frozenset({("security", 0)}),
    # 'Its price is 12 euros' needs the plan name, four sentences earlier.
    "q07": frozenset({("team-plan", 0), ("team-plan", 4)}),
    "q08": frozenset({("team-plan", 3)}),
    "q09": frozenset({("retention", 2)}),
    "q10": frozenset({("retention", 0)}),
}

# What each query was written to probe. Descriptive only; not used in scoring.
PURPOSE: dict[str, str] = {
    "q01": "boundary context (adjacent antecedent) + FAQ sentence that restates the question",
    "q02": "boundary context (adjacent antecedent)",
    "q03": "exact identifier next to a near-identical identifier",
    "q04": "synonyms with no lexical overlap",
    "q05": "partial lexical overlap",
    "q06": "easy control query",
    "q07": "boundary context (antecedent four sentences away) + competing price sentence",
    "q08": "lexical anchor plus synonym",
    "q09": "lexical anchor plus synonym",
    "q10": "vocabulary gap that no channel covers (candidate ceiling)",
}
