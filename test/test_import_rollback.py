"""Tests for atomic import rollback.

When a file operation (move/copy/link/write) fails partway through
importing an album — e.g. because the disk is full — the whole task
must be rolled back: database rows, file locations and files scheduled
for deletion all return to their pre-import state. Rerunning the
import afterwards must produce exactly the same result as an
uninterrupted import.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from beets.importer.tasks import ImportTaskFactory
from beets.library import Item
from beets.test import _common
from beets.test.helper import AutotagImportHelper, AutotagStub, ImportHelper

# Fields that legitimately differ between two separate imports.
VOLATILE_FIELDS = {"added", "mtime"}

TRACK_COUNT = 3


def db_state(lib, ignore=frozenset()):
    """Return a comparable snapshot of every row in the library."""

    def dump(model):
        return {k: model[k] for k in model if k not in ignore}

    return {
        "items": sorted((dump(i) for i in lib.items()), key=lambda d: d["id"]),
        "albums": sorted(
            (dump(a) for a in lib.albums()), key=lambda d: d["id"]
        ),
    }


def fs_state(**roots: Path) -> dict[tuple[str, str], str]:
    """Return a comparable snapshot of the directory trees under `roots`:
    every file (by content hash) and every directory.
    """
    state = {}
    for label, root in roots.items():
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames:
                rel = str((Path(dirpath) / name).relative_to(root))
                state[(label, rel)] = "dir"
            for name in filenames:
                path = Path(dirpath) / name
                rel = str(path.relative_to(root))
                state[(label, rel)] = hashlib.sha1(
                    path.read_bytes()
                ).hexdigest()
    return state


@contextmanager
def fail_nth_call(cls, attr, n):
    """Patch `cls.attr` to raise `OSError` (simulating a full disk)
    when it is called for the `n`-th time (1-based).
    """
    original = getattr(cls, attr)
    calls = []

    def wrapper(*args, **kwargs):
        calls.append(1)
        if len(calls) == n:
            raise OSError(28, "No space left on device")
        return original(*args, **kwargs)

    with patch.object(cls, attr, wrapper):
        yield


# Matches the simulated disk-full error raised by `fail_nth_call`.
DISK_FULL = "No space left on device"


@contextmanager
def bitrate_overrides(bitrates_by_title):
    """Force specific per-title bitrates on newly-read import items.

    The test mp3 fixtures all share one real bitrate, so this patches
    `ImportTaskFactory.read_item` to simulate different-quality
    encodes without needing distinct binary fixtures.
    """
    original = ImportTaskFactory.read_item

    def patched(self, path):
        item = original(self, path)
        if item is not None and item.title in bitrates_by_title:
            item.bitrate = bitrates_by_title[item.title]
        return item

    with patch.object(ImportTaskFactory, "read_item", patched):
        yield


class TestImportRollback(ImportHelper):
    db_on_disk = True

    def setup_beets(self):
        super().setup_beets()
        self.track_paths = self.prepare_album_for_import(TRACK_COUNT)

    def run_importer(self, **kwargs):
        import_dir = kwargs.pop("import_dir", None)
        importer = self.setup_importer(import_dir, autotag=False, **kwargs)
        importer.run()
        return importer

    def assert_pristine(self, before_db, before_fs):
        """The library and the import directory are untouched and no
        rollback trash is left behind.
        """
        assert db_state(self.lib) == before_db
        assert fs_state(import_dir=self.import_path, lib=self.lib_path) == (
            before_fs
        )
        assert not list(self.lib_path.glob(".beets-rollback-*"))

    def assert_import_succeeded(self, moved=False):
        """The state after a successful as-is import of the album."""
        assert len(self.lib.albums()) == 1
        items = self.lib.items()
        assert len(items) == TRACK_COUNT
        for item in items:
            assert item.filepath.exists()
            assert item.filepath.is_relative_to(self.lib_path)
        for track_path in self.track_paths:
            assert track_path.exists() != moved
        assert not list(self.lib_path.glob(".beets-rollback-*"))

    def snapshot(self):
        return (
            db_state(self.lib),
            fs_state(import_dir=self.import_path, lib=self.lib_path),
        )

    # Failure injection.

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"copy": True}, id="copy"),
            pytest.param({"move": True}, id="move"),
            pytest.param(
                {"link": True},
                id="link",
                marks=pytest.mark.skipif(
                    not _common.HAVE_SYMLINK, reason="need symlinks"
                ),
            ),
            pytest.param(
                {"hardlink": True},
                id="hardlink",
                marks=pytest.mark.skipif(
                    not _common.HAVE_HARDLINK, reason="need hardlinks"
                ),
            ),
            pytest.param({"reflink": "auto"}, id="reflink_auto"),
        ],
    )
    def test_file_move_failure_rolls_back(self, config):
        """Fail the 2nd file operation of {}: everything is restored."""
        before_db, before_fs = self.snapshot()

        with (
            fail_nth_call(Item, "move", 2),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer(**config)

        self.assert_pristine(before_db, before_fs)

        # Rerunning the import now succeeds completely.
        self.run_importer(**config)
        self.assert_import_succeeded(moved=config.get("move", False))

    def test_threaded_file_move_failure_rolls_back(self):
        self.config["threaded"] = True
        before_db, before_fs = self.snapshot()

        with (
            fail_nth_call(Item, "move", 2),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer()

        self.assert_pristine(before_db, before_fs)

        self.run_importer()
        self.assert_import_succeeded()

    def test_singleton_failure_rolls_back(self):
        """A failing singleton task rolls back itself; tasks that
        completed before it stay imported.
        """
        self.config["import"]["duplicate_action"] = "remove"

        with (
            fail_nth_call(Item, "move", 2),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer(singletons=True)

        # The first singleton was fully imported before the failure.
        assert len(self.lib.items()) == 1
        item = self.lib.items().get()
        assert item.filepath.exists()
        # Everything else is untouched.
        remaining_fs = fs_state(import_dir=self.import_path, lib=self.lib_path)
        assert ("import_dir", "album/track_1.mp3") in remaining_fs
        assert ("import_dir", "album/track_2.mp3") in remaining_fs
        assert ("import_dir", "album/track_3.mp3") in remaining_fs
        assert not list(self.lib_path.glob(".beets-rollback-*"))

        self.run_importer(singletons=True)
        assert len(self.lib.items()) == TRACK_COUNT
        for item in self.lib.items():
            assert item.filepath.exists()

    def test_duplicate_remove_failure_rolls_back(self):
        """With duplicate_action=remove, a failure must bring back the
        old duplicate's rows and files instead of losing them.
        """
        album = self.add_album_fixture()
        album.albumartist = "Tag Artist"
        album.album = "Tag Album"
        album.store()
        old_item_paths = [i.filepath for i in album.items()]
        assert all(p.exists() for p in old_item_paths)

        self.config["import"]["duplicate_action"] = "remove"
        before_db, before_fs = self.snapshot()

        with (
            fail_nth_call(Item, "move", 1),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer()

        self.assert_pristine(before_db, before_fs)

        # Rerun: the duplicate is replaced by the new album.
        self.run_importer()
        assert len(self.lib.albums()) == 1
        assert len(self.lib.items()) == TRACK_COUNT
        assert all(not p.exists() for p in old_item_paths)
        self.assert_import_succeeded()

    def test_duplicate_upgrade_failure_rolls_back(self):
        """With duplicate_action=upgrade, a failure must restore the
        superseded old tracks and their album.
        """
        old_item = self.add_item_fixture(
            artist="Tag Artist",
            albumartist="Tag Artist",
            album="Tag Album",
            title="Tag Track 1",
            bitrate=64000,
        )
        old_album = self.lib.add_album([old_item])
        old_item_path = old_item.filepath
        assert old_item_path.exists()

        self.config["import"]["duplicate_action"] = "upgrade"
        before_db, before_fs = self.snapshot()

        with (
            bitrate_overrides({"Tag Track 1": 320000}),
            fail_nth_call(Item, "move", 1),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer()

        self.assert_pristine(before_db, before_fs)
        assert self.lib.get_item(old_item.id) is not None
        assert self.lib.get_album(old_album.id) is not None

        # Rerun: the upgrade goes through.
        with bitrate_overrides({"Tag Track 1": 320000}):
            self.run_importer()
        assert self.lib.get_item(old_item.id) is None
        assert not old_item_path.exists()
        assert len(self.lib.items()) == TRACK_COUNT
        for item in self.lib.items():
            assert item.filepath.exists()

    def test_reimport_failure_restores_replaced_rows(self):
        """Reimporting files that are already in the library replaces
        their rows; a failure must restore the original rows.
        """
        self.run_importer()
        before_db, before_fs = (db_state(self.lib), fs_state(lib=self.lib_path))

        # Reimport the library files themselves: their rows are replaced.
        with (
            fail_nth_call(Item, "move", 2),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer(import_dir=self.lib_path)

        assert db_state(self.lib) == before_db
        assert fs_state(lib=self.lib_path) == before_fs
        assert not list(self.lib_path.glob(".beets-rollback-*"))

    @pytest.mark.parametrize(
        "config", [{"incremental": True}, {"resume": True}]
    )
    def test_failed_task_is_not_recorded_as_done(self, config):
        """Incremental/resume state is only written for completed tasks,
        so a failed album is picked up again on the next run.
        """
        with (
            fail_nth_call(Item, "move", 2),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer(**config)

        self.run_importer(**config)
        self.assert_import_succeeded()

    def test_rerun_matches_clean_import(self):
        """A failed-then-rerun import produces exactly the result of an
        uninterrupted import.
        """
        # Uninterrupted baseline.
        self.run_importer()
        clean_db = db_state(self.lib, ignore=VOLATILE_FIELDS)
        clean_fs = fs_state(lib=self.lib_path)

        # Reset the library to its empty state.
        with self.lib.transaction() as tx:
            for table in (
                "items",
                "albums",
                "item_attributes",
                "album_attributes",
            ):
                tx.mutate(f"DELETE FROM {table}")
        self.lib._memotable = {}
        self.lib._group_memotable = {}
        shutil.rmtree(self.lib_path)
        self.lib_path.mkdir()

        # Fail, then rerun.
        with (
            fail_nth_call(Item, "move", 2),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.run_importer()
        self.run_importer()

        assert db_state(self.lib, ignore=VOLATILE_FIELDS) == clean_db
        assert fs_state(lib=self.lib_path) == clean_fs


class TestImportRollbackOnWrite(AutotagImportHelper):
    """Tag writes only happen for applied matches, so these use the
    autotag stub rather than an as-is import.
    """

    matching = AutotagStub.IDENT
    db_on_disk = True

    def setup_beets(self):
        super().setup_beets()
        self.track_paths = self.prepare_album_for_import(TRACK_COUNT)

    def test_tag_write_failure_rolls_back(self):
        before_db = db_state(self.lib)
        before_fs = fs_state(import_dir=self.import_path, lib=self.lib_path)

        with (
            fail_nth_call(Item, "write", 2),
            pytest.raises(OSError, match=DISK_FULL),
        ):
            self.setup_importer().run()

        assert db_state(self.lib) == before_db
        assert fs_state(import_dir=self.import_path, lib=self.lib_path) == (
            before_fs
        )
        assert not list(self.lib_path.glob(".beets-rollback-*"))

        # Rerunning the import now succeeds completely.
        self.setup_importer().run()
        assert len(self.lib.albums()) == 1
        assert len(self.lib.items()) == TRACK_COUNT
        for item in self.lib.items():
            assert item.filepath.exists()
        for track_path in self.track_paths:
            assert track_path.exists()
