"""Instance schema for a searching agent: what it is told about the benchmark and the search."""

from __future__ import annotations

from pydantic import BaseModel


class SearchTask(BaseModel):
    benchmark_name: str
    task_description: str
    candidate_definition: str
    criteria: str = ""
    identity_fields: list[str]
    identity_normalize: dict[str, str] = {}
    data_source: str
    admissibility: list[dict] = []
    constraints: str = ""
    spread: str = ""
    sample_size: int = 100
    known_from: list[str] = []  # stock files or run dirs whose candidates do not count again
    territory: str = ""  # this attempt's part of the search space, one sentence
    sibling_territories: list[str] = []  # the parts the parallel attempts cover
