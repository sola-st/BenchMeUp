"""Instance schema for a worker: what it is told about the benchmark and its candidates."""

from __future__ import annotations

from pydantic import BaseModel


class WorkerTask(BaseModel):
    benchmark_name: str
    task_description: str
    candidate_definition: str
    source_requirements: str
    criteria: str
    sample_size: int = 10
    criteria_list: list[str] = []
    quality_gates_list: list[str] = []
    instance_format: str = ""
    examples: list[dict] = []
    admissibility: list[dict] = []
    identity_field: str = ""
    identity_fields: list[str] = []  # what the search hands over
    fields_to_establish: list[str] = []  # instance.json fields the worker establishes itself
    candidates: list[dict] = []  # the pool the system hands to the worker, one at a time
