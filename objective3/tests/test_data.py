import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from content.pkt.recommendation.data import inspect_archive, iter_histories


def record(user="alice", ast="[[4,[5,[-1]]]]", problems="10;11"):
    return "\n".join((f"{user},2", problems, "[1,2];[2]", "", f"{ast};[]", "", "1;0", ""))


class DataTests(unittest.TestCase):
    def archive(self, contents, ending="\r\n"):
        temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "fixture.zip"
        with ZipFile(path, "w") as zf:
            zf.writestr("astnn/astnn_train1", ending.join(contents))
        return path

    def multi_split_archive(self, splits):
        temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "fixture.zip"
        names = {"train1": "astnn/astnn_train1", "valid1": "astnn/astnn_valid1", "test": "astnn/astnn_test"}
        with ZipFile(path, "w") as zf:
            for split, user, problem in splits:
                zf.writestr(names[split], record(user=user, problems=f"{problem};{problem + 1}"))
        return path

    def test_identity_schema_and_optional_final_separator(self):
        path = self.archive([record()])
        history, = iter_histories(path)
        self.assertEqual((history.learner_id, history.source_split), ("alice", "train1"))
        first, second = history.interactions
        self.assertEqual((first.sequence_index, first.problem_id, first.concept_ids, first.correct), (0, 10, (1, 2), 1))
        self.assertEqual(first.ast_roots, [[4, [5, [-1]]]])
        self.assertEqual((second.sequence_index, second.problem_id, second.correct), (1, 11, 0))
        self.assertEqual(second.ast_roots, [])

    def test_final_record_without_trailing_newline(self):
        path = self.archive([record().rstrip("\n")])
        history, = iter_histories(path)
        self.assertEqual(len(history.interactions), 2)

    def test_bad_count_and_ast_raise_contextual_errors(self):
        bad_count = record().replace("10;11", "10")
        with self.assertRaisesRegex(ValueError, "train1.*record 1 line 2"):
            list(iter_histories(self.archive([bad_count])))
        bad_ast = record(ast="[[4,5]]")
        with self.assertRaisesRegex(ValueError, "train1\\) record 1 interaction 1 line 5"):
            list(iter_histories(self.archive([bad_ast])))

    def test_duplicate_learner_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate learner_id"):
            list(iter_histories(self.archive([record(), record()])))

    def test_ast_checked_when_not_returned(self):
        with self.assertRaisesRegex(ValueError, "malformed AST"):
            list(iter_histories(self.archive([record(ast="[42]")]), include_asts=False))

    def test_positive_problem_ids_and_nonnegative_tokens(self):
        with self.assertRaisesRegex(ValueError, "problem IDs must be positive"):
            list(iter_histories(self.archive([record(problems="0;11")])) )
        with self.assertRaisesRegex(ValueError, "malformed AST"):
            list(iter_histories(self.archive([record(ast="[[-2]]")])) )

    def test_sentinel_requires_legacy_child_shape(self):
        self.assertEqual(len(next(iter_histories(self.archive([record()]))).interactions), 2)
        for ast in ("[-1]", "[[4,-1]]", "[[4,[-1,3]]]"):
            with self.subTest(ast=ast), self.assertRaisesRegex(ValueError, "malformed|sentinel"):
                list(iter_histories(self.archive([record(ast=ast)])))

    def test_one_trailing_semicolon_is_accepted_but_interior_empty_is_not(self):
        self.assertEqual(len(next(iter_histories(self.archive([record(problems="10;11;")]))).interactions), 2)
        with self.assertRaisesRegex(ValueError, "empty entry"):
            list(iter_histories(self.archive([record(problems="10;;11")])) )

    def test_inspector_profiles_small_archive_and_reports_user_overlaps(self):
        path = self.multi_split_archive((
            ("train1", "alice", 10),
            ("valid1", "bob", 20),
            ("test", "carol", 10),
        ))
        profile = inspect_archive(path)
        self.assertEqual(profile["union"], {"learners": 3, "interactions": 6, "distinct_problems": 4, "concepts": 2})
        self.assertEqual(profile["user_overlaps"]["train1/valid1"], 0)
        self.assertEqual(profile["problem_overlaps"]["train1/test"], 2)


if __name__ == "__main__":
    unittest.main()
