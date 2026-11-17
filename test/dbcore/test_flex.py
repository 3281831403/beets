"""Tests for automatic typing of undeclared flexible attributes.

Undeclared flexible attributes (i.e. those not listed in ``types:``) should
"speak by value": numbers compare and sort numerically, booleans are
queryable as true/false, lists can be stored, read back and queried by
element, and plain text keeps its substring query semantics and original
display. Declared types and fixed fields are exercised elsewhere and must be
unaffected.
"""

import pytest

from beets.test.helper import IOMixin, TestHelper

_p = pytest.param


@pytest.fixture(scope="class")
def helper(class_helper):
    return class_helper


class TestFlexNumeric:
    """Numbers written as native values compare and sort numerically."""

    @pytest.fixture(scope="class")
    @classmethod
    def lib(cls, helper):
        for rating in (1, 10, 2, 4, 14):
            helper.add_item(title=f"num-{rating}", rating=rating)
        return helper.lib

    def test_sort_is_numeric(self, lib):
        assert [i.title for i in lib.items("rating+")] == [
            "num-1",
            "num-2",
            "num-4",
            "num-10",
            "num-14",
        ]

    def test_range_query(self, lib):
        assert {i.title for i in lib.items("rating:8..10")} == {"num-10"}

    def test_open_range_query(self, lib):
        assert {i.title for i in lib.items("rating:..5")} == {
            "num-1",
            "num-2",
            "num-4",
        }

    def test_exact_does_not_substring_match(self, lib):
        assert {i.title for i in lib.items("rating:4")} == {"num-4"}


class TestFlexNumericText:
    """Pure-numeric text (as written by the command line or found in old
    libraries) is treated as numeric as well.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def lib(cls, helper):
        for rating in ("1", "10", "2", "4", "14"):
            helper.add_item(title=f"str-{rating}", rating=rating)
        return helper.lib

    def test_sort_is_numeric(self, lib):
        assert [i.title for i in lib.items("rating+")] == [
            "str-1",
            "str-2",
            "str-4",
            "str-10",
            "str-14",
        ]

    def test_range_query(self, lib):
        assert {i.title for i in lib.items("rating:8..10")} == {"str-10"}

    def test_open_range_query(self, lib):
        assert {i.title for i in lib.items("rating:..5")} == {
            "str-1",
            "str-2",
            "str-4",
        }

    def test_exact_does_not_substring_match(self, lib):
        assert {i.title for i in lib.items("rating:4")} == {"str-4"}


class TestFlexBoolean:
    """Booleans can be queried with true/false."""

    @pytest.fixture(scope="class")
    @classmethod
    def lib(cls, helper):
        helper.add_item(title="live-yes", live=True)
        helper.add_item(title="live-no", live=False)
        return helper.lib

    def test_true_query(self, lib):
        assert {i.title for i in lib.items("live:true")} == {"live-yes"}

    def test_false_query(self, lib):
        assert {i.title for i in lib.items("live:false")} == {"live-no"}

    def test_native_boolean_round_trips(self, lib):
        # Native booleans are stored as the numeric strings "1"/"0" but
        # still query as true/false.
        assert lib.items("title:live-yes").get().live == "1"
        assert lib.items("title:live-no").get().live == "0"


class TestFlexList:
    """Lists can be stored, read back, and queried by element."""

    @pytest.fixture(scope="class")
    @classmethod
    def lib(cls, helper):
        helper.add_item(title="moody", moods=["happy", "sad"])
        helper.add_item(title="single", moods=["lonely"])
        return helper.lib

    def test_read_back(self, lib):
        assert lib.items("title:moody").get().moods == ["happy", "sad"]

    def test_single_element_read_back(self, lib):
        assert lib.items("title:single").get().moods == ["lonely"]

    def test_query_by_element(self, lib):
        assert {i.title for i in lib.items("moods:happy")} == {"moody"}
        assert {i.title for i in lib.items("moods:sad")} == {"moody"}
        assert {i.title for i in lib.items("moods:lonely")} == {"single"}

    def test_query_missing_element(self, lib):
        assert lib.items("moods:angry").get() is None


class TestFlexMixedColumn:
    """A column mixing numbers and text has a stable total order: missing
    values first, then numbers, then text.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def lib(cls, helper):
        helper.add_item(title="missing")
        helper.add_item(title="num-2", rating=2)
        helper.add_item(title="num-10", rating=10)
        helper.add_item(title="num-4", rating=4)
        helper.add_item(title="text", rating="great")
        return helper.lib

    def test_ascending_total_order(self, lib):
        assert [i.title for i in lib.items("rating+")] == [
            "missing",
            "num-2",
            "num-4",
            "num-10",
            "text",
        ]

    def test_descending_total_order(self, lib):
        assert [i.title for i in lib.items("rating-")] == [
            "text",
            "num-10",
            "num-4",
            "num-2",
            "missing",
        ]


class TestFlexText:
    """Plain-text rows keep substring matching and their original display."""

    @pytest.fixture(scope="class")
    @classmethod
    def lib(cls, helper):
        helper.add_item(title="note", note="hello world")
        return helper.lib

    def test_substring_query(self, lib):
        assert {i.title for i in lib.items("note:hello")} == {"note"}
        assert {i.title for i in lib.items("note:world")} == {"note"}

    def test_original_display(self, lib):
        item = lib.items("note:hello").get()
        assert item.formatted().get("note") == "hello world"


class TestFlexModifyCommand(IOMixin, TestHelper):
    """The ``modify`` command writes strings, which are still numeric or
    boolean when queried.
    """

    def modify(self, *args: str) -> None:
        self.run_command("modify", "--yes", "--nowrite", "--nomove", *args)

    def test_modify_numeric_text_is_numeric(self):
        item = self.add_item(title="mod-num")
        self.modify("rating=7")

        item.load()
        assert item.rating == "7"
        assert {i.title for i in self.lib.items("rating:7..9")} == {"mod-num"}

    def test_modify_boolean_text_is_boolean(self):
        item = self.add_item(title="mod-bool")
        self.modify("live=true")

        item.load()
        assert item.live == "true"
        assert {i.title for i in self.lib.items("live:true")} == {"mod-bool"}
