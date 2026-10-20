"""Validated, streaming reader for the ASTNN Codeforces ZIP records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile


_SPLITS = {"train": "train1", "valid": "valid1", "validation": "valid1", "test": "test"}
_FILES = {"train1": ("astnn/astnn_train1",), "valid1": ("astnn/astnn_valid1",), "test": ("astnn/astnn_test",)}


@dataclass(frozen=True)
class Interaction:
    learner_id: str
    sequence_index: int
    problem_id: int
    concept_ids: tuple[int, ...]
    correct: int
    ast_roots: list | None
    source_split: str
    source_record: int


@dataclass(frozen=True)
class LearnerHistory:
    learner_id: str
    interactions: tuple[Interaction, ...]
    source_split: str


def _ast_valid(roots, where):
    if not isinstance(roots, list):
        raise ValueError(f"{where}: AST roots must be a JSON list")
    stack = [(node, True) for node in roots]
    while stack:
        node, is_root = stack.pop()
        if not isinstance(node, list) or not node:
            raise ValueError(f"{where}: malformed AST node {node!r}")
        if node[0] == -1:
            if is_root or node != [-1]:
                raise ValueError(f"{where}: sentinel must be exactly [-1] as a child")
            continue
        if not isinstance(node[0], int) or isinstance(node[0], bool) or node[0] < 0:
            raise ValueError(f"{where}: malformed AST node {node!r}")
        for child in node[1:]:
            if not isinstance(child, list):
                raise ValueError(f"{where}: malformed AST child {child!r}")
            stack.append((child, False))


def _json(text, where):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{where}: invalid JSON: {exc}") from exc


def _parse_record(lines, split, member, record_no, include_asts, verify_asts):
    where = f"{member} ({split}) record {record_no}"
    if len(lines) != 8:
        raise ValueError(f"{where}: expected 8 lines, got {len(lines)}")
    header = lines[0].split(",")
    if len(header) != 2 or not header[0]:
        raise ValueError(f"{where} line 1: expected learner_id,count")
    learner, count_text = header
    try:
        count = int(count_text)
    except ValueError as exc:
        raise ValueError(f"{where} line 1: invalid interaction count") from exc
    if count < 0:
        raise ValueError(f"{where} line 1: negative interaction count")
    if lines[3] or lines[5] or lines[7]:
        raise ValueError(f"{where}: lines 4, 6, and 8 must be blank")

    def fields(number):
        raw_line = lines[number - 1]
        vals = raw_line.split(";") if raw_line else []
        if vals and vals[-1] == "":
            vals.pop()  # The legacy writer sometimes appends one final delimiter.
        if any(value == "" for value in vals):
            raise ValueError(f"{where} line {number}: empty entry")
        if len(vals) != count:
            raise ValueError(f"{where} line {number}: expected {count} entries, got {len(vals)}")
        return vals
    problem_values = fields(2)
    try:
        problems = [int(x) for x in problem_values]
    except ValueError as exc:
        raise ValueError(f"{where} line 2: problem IDs must be integers") from exc
    if any(problem_id <= 0 for problem_id in problems):
        raise ValueError(f"{where} line 2: problem IDs must be positive")
    concepts = []
    for i, raw in enumerate(fields(3), 1):
        tags = _json(raw, f"{where} interaction {i} line 3")
        if not isinstance(tags, list) or not tags or any(not isinstance(x, int) or isinstance(x, bool) or not 1 <= x <= 37 for x in tags):
            raise ValueError(f"{where} interaction {i}: concept_ids must be nonempty integers in 1..37")
        concepts.append(tuple(sorted(set(tags))))
    asts = []
    for i, raw in enumerate(fields(5), 1):
        roots = _json(raw, f"{where} interaction {i} line 5")
        if verify_asts:
            _ast_valid(roots, f"{where} interaction {i} line 5")
        asts.append(roots if include_asts else None)
    label_values = fields(7)
    try:
        labels = [int(x) for x in label_values]
    except ValueError as exc:
        raise ValueError(f"{where} line 7: labels must be integers") from exc
    if any(x not in (0, 1) for x in labels):
        raise ValueError(f"{where} line 7: labels must be binary")
    result = []
    for i, (pid, tags, label, roots) in enumerate(zip(problems, concepts, labels, asts)):
        result.append(Interaction(learner, i, pid, tags, label, roots, split, record_no))
    return LearnerHistory(learner, tuple(result), split)


def _members(zf, split):
    normalized = {n.lower().replace("\\", "/"): n for n in zf.namelist()}
    for candidate in _FILES[split]:
        for name, original in normalized.items():
            if name.rstrip("/").endswith(candidate):
                return original
    raise ValueError(f"archive: missing ASTNN {split} member")


def _iter_member(zf, member, split, include_asts=True, verify_asts=True):
    seen = set()
    with zf.open(member) as stream:
        record, lines = 0, []
        for raw in stream:
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{member} record {record + 1} line {len(lines) + 1}: invalid UTF-8") from exc
            if not lines and not line:
                continue
            lines.append(line)
            if len(lines) == 8:
                if lines[0]:
                    record += 1
                    history = _parse_record(lines, split, member, record, include_asts, verify_asts)
                    if history.learner_id in seen:
                        raise ValueError(f"{member} ({split}) record {record}: duplicate learner_id {history.learner_id!r}")
                    seen.add(history.learner_id)
                    yield history
                elif any(lines):
                    raise ValueError(f"{split} record {record + 1}: malformed separator")
                lines = []
        if lines:
            if len(lines) == 1 and not lines[0]:
                return
            if len(lines) == 7 and lines[6]:
                record += 1
                history = _parse_record(lines + [""], split, member, record, include_asts, verify_asts)
                if history.learner_id in seen:
                    raise ValueError(f"{member} ({split}) record {record}: duplicate learner_id {history.learner_id!r}")
                yield history
                return
            if len(lines) == 8 and not any(lines):
                return
            raise ValueError(f"{split} record {record + 1}: incomplete 8-line record ({len(lines)} lines)")


def iter_histories(zip_path, split="train", include_asts=True, verify_asts=True) -> Iterator[LearnerHistory]:
    """Yield validated learner histories from one split without extracting the ZIP."""
    key = str(split).lower()
    canonical = _SPLITS.get(key, key)
    if canonical not in _FILES:
        raise ValueError(f"split must be one of train, valid, test; got {split!r}")
    with ZipFile(zip_path) as zf:
        member = _members(zf, canonical)
        yield from _iter_member(zf, member, canonical, include_asts, verify_asts)


def inspect_archive(zip_path):
    """Return a JSON-safe, validation-backed profile of the three ASTNN splits."""
    path = Path(zip_path)
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    result = {"archive": str(path), "sha256": checksum.hexdigest(), "splits": {}, "user_overlaps": {}, "problem_overlaps": {}, "union": {}}
    users_by_split, problems_by_split, all_tags, problem_tags = {}, {}, set(), {}
    for split in ("train1", "valid1", "test"):
        profile = {"learners": 0, "interactions": 0, "problems": set(), "correct": 0, "incorrect": 0, "min_history": None, "max_history": 0, "ast_roots": 0, "ast_nodes": 0, "ast_sentinels": 0, "token_min": None, "token_max": None, "ast_roots_over_30": 0, "histories_over_200": 0}
        split_users = set()
        with ZipFile(path) as zf:
            member = _members(zf, split)
            histories = _iter_member(zf, member, split, True, True)
            for h in histories:
                profile["learners"] += 1
                n = len(h.interactions)
                profile["interactions"] += n
                profile["min_history"] = n if profile["min_history"] is None else min(profile["min_history"], n)
                profile["max_history"] = max(profile["max_history"], n)
                profile["histories_over_200"] += int(n > 200)
                split_users.add(h.learner_id)
                for x in h.interactions:
                    profile["problems"].add(x.problem_id); all_tags.update(x.concept_ids)
                    previous = problem_tags.setdefault(x.problem_id, x.concept_ids)
                    if previous != x.concept_ids:
                        raise ValueError(f"{split} record {x.source_record} interaction {x.sequence_index + 1}: problem {x.problem_id} has inconsistent concept_ids")
                    profile["correct" if x.correct else "incorrect"] += 1
                    profile["ast_roots"] += len(x.ast_roots)
                    profile["ast_roots_over_30"] += int(len(x.ast_roots) > 30)
                    stack = list(x.ast_roots)
                    while stack:
                        node = stack.pop()
                        if node[0] == -1:
                            profile["ast_sentinels"] += 1
                        else:
                            tok = node[0]; profile["ast_nodes"] += 1
                            profile["token_min"] = tok if profile["token_min"] is None else min(profile["token_min"], tok)
                            profile["token_max"] = tok if profile["token_max"] is None else max(profile["token_max"], tok)
                            stack.extend(node[1:])
        users_by_split[split] = split_users
        problems_by_split[split] = profile["problems"]
        profile["distinct_problems"] = len(profile.pop("problems"))
        result["splits"][split] = profile
    for a, b in (("train1", "valid1"), ("train1", "test"), ("valid1", "test")):
        overlap = users_by_split[a] & users_by_split[b]
        if overlap:
            raise ValueError(f"learner IDs overlap between {a} and {b}: {sorted(overlap)[:3]}")
        result["user_overlaps"][f"{a}/{b}"] = len(overlap)
        result["problem_overlaps"][f"{a}/{b}"] = len(problems_by_split[a] & problems_by_split[b])
    result["union"] = {"learners": sum(len(v) for v in users_by_split.values()), "interactions": sum(v["interactions"] for v in result["splits"].values()), "distinct_problems": len(set.union(*problems_by_split.values())), "concepts": len(all_tags)}
    return result
