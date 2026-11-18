"""Rollback support for import tasks.

Importing an album is not naturally atomic: the importer adds rows to
the library database, moves/copies/links the files into the library,
writes their tags and deletes superseded duplicates as separate steps.
If any of these steps fails partway (a full disk being the classic
example), the library used to be left in an inconsistent state: some
files landed in the library while their source disappeared, database
rows pointed at files that were never moved, and old duplicates were
already deleted.

:class:`ImportRollback` is a per-task journal that records every
destructive operation performed while importing one task, so that a
failure anywhere in the file-manipulation phase can be undone: files
that were moved are moved back, files that were created are removed,
files that were deleted are restored from a temporary trash directory
inside the library, and database rows are returned to their pre-import
state. Once the task completes successfully, :meth:`ImportRollback.commit`
discards the journal and permanently deletes the trashed files.
"""

from __future__ import annotations

import logging
import os
import shutil
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any

from beets import config, util

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from beets.dbcore import Model
    from beets.library import Album, Item, Library

log = logging.getLogger("beets")

# Filesystem journal entry kinds.
_MOVE = "move"  # (src, dest): a file was moved; undo moves it back.
_CREATE = "create"  # (dest,): a file was created; undo removes it.
_TRASH = "trash"  # (trash_path, orig): a file was "deleted" into the
# trash directory; undo moves it back.


class ImportRollback:
    """Journal of one import task's destructive filesystem and database
    operations.

    The journal is filled as the import proceeds and is either
    *committed* (the task finished; trashed files are deleted for good)
    or *rolled back* (something failed; every recorded operation is
    undone so the library and the source files return to their
    pre-import state).
    """

    def __init__(self) -> None:
        # Filesystem operations, in the order they were performed.
        self._fs_log: list[tuple[str, Any, Any]] = []
        # Snapshots (copies) of database rows deleted by the task.
        self._removed_items: list[Item] = []
        self._removed_albums: list[Album] = []
        # Database rows added by the task.
        self._added_items: list[Item] = []
        self._added_albums: list[Album] = []
        # Per-task directory holding "deleted" files, created lazily.
        self._trash_dir: str | None = None
        self._trash_count = 0
        # Set once the journal has been committed or rolled back.
        self._done = False

    # Recording database changes.

    def record_removed(
        self, items: Iterable[Item] = (), albums: Iterable[Album] = ()
    ) -> None:
        """Snapshot rows that are about to be deleted from the library.

        Must be called *before* the rows are deleted; the models are
        copied so later mutations cannot corrupt the snapshot.
        """
        self._removed_items.extend(item.copy() for item in items)
        self._removed_albums.extend(album.copy() for album in albums)

    def record_added(
        self, items: Iterable[Item] = (), albums: Iterable[Album] = ()
    ) -> None:
        """Record rows that were added to the library by the task.

        Only the models' ids are needed, so the live objects can be
        kept directly.
        """
        self._added_items.extend(items)
        self._added_albums.extend(albums)

    # Recording filesystem changes.

    def file_moved(self, src: util.PathLike, dest: util.PathLike) -> None:
        """Record that a file was moved from `src` to `dest`."""
        self._fs_log.append((_MOVE, src, dest))

    def file_created(self, dest: util.PathLike) -> None:
        """Record that a file was created at `dest` (copy/link)."""
        self._fs_log.append((_CREATE, dest, None))

    def trash(self, path: util.PathLike, lib: Library) -> None:
        """ "Delete" a file recoverably by moving it into the trash dir.

        The file disappears from its original location immediately (so
        subsequent operations see the same state as with a real delete),
        but it can be put back until :meth:`commit` is called.
        """
        if self._trash_dir is None:
            self._trash_dir = mkdtemp(
                prefix=".beets-rollback-", dir=os.fsdecode(lib.directory)
            )
        self._trash_count += 1
        basename = os.path.basename(os.fsdecode(path))
        trash_path = os.path.join(
            self._trash_dir, f"{self._trash_count}_{basename}"
        )
        util.move(path, trash_path)
        self._fs_log.append((_TRASH, trash_path, path))

    # Finishing.

    def commit(self) -> None:
        """The task succeeded: delete trashed files and drop the journal."""
        self._done = True
        self._fs_log.clear()
        self._removed_items.clear()
        self._removed_albums.clear()
        self._added_items.clear()
        self._added_albums.clear()
        self._empty_trash()

    def rollback(self, lib: Library) -> None:
        """Undo every recorded operation, restoring the pre-import state.

        Best effort: individual steps that fail are logged and skipped
        so as much state as possible is restored; the exception that
        triggered the rollback is never masked.
        """
        if self._done:
            return
        self._done = True

        for kind, first, second in reversed(self._fs_log):
            try:
                self._undo_file_op(lib, kind, first, second)
            except Exception as exc:
                log.error(
                    "rollback: could not restore {}: {}",
                    util.displayable_path(second if kind == _TRASH else first),
                    exc,
                )
        self._fs_log.clear()
        self._empty_trash()

        try:
            with lib.transaction() as tx:
                for model in [*self._added_items, *self._added_albums]:
                    self._delete_row(tx, model)
                for model in [*self._removed_albums, *self._removed_items]:
                    self._restore_row(tx, model)
        except Exception as exc:
            log.error("rollback: could not restore the database: {}", exc)
        self._removed_items.clear()
        self._removed_albums.clear()
        self._added_items.clear()
        self._added_albums.clear()
        # Invalidate cached template data (e.g. %aunique{}) that raw SQL
        # bypassed.
        lib._memotable = {}
        lib._group_memotable = {}

    # Helpers.

    def _empty_trash(self) -> None:
        if self._trash_dir is not None:
            shutil.rmtree(self._trash_dir, ignore_errors=True)
            self._trash_dir = None

    def _undo_file_op(
        self, lib: Library, kind: str, first: Any, second: Any
    ) -> None:
        clutter: Sequence[str] = config["clutter"].as_str_seq()
        if kind == _MOVE:
            src, dest = first, second
            if os.path.exists(util.syspath(dest)):
                util.mkdirall(src)
                util.move(dest, src)
            util.prune_dirs(
                os.path.dirname(dest), lib.directory, clutter=clutter
            )
        elif kind == _CREATE:
            dest = first
            util.remove(dest)
            util.prune_dirs(
                os.path.dirname(dest), lib.directory, clutter=clutter
            )
        elif kind == _TRASH:
            trash_path, orig = first, second
            if os.path.exists(util.syspath(trash_path)):
                util.mkdirall(orig)
                util.move(trash_path, orig)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown rollback operation {kind!r}")

    @staticmethod
    def _delete_row(tx: Any, model: Model) -> None:
        """Delete a row previously added by the task (by id)."""
        if model.id is None:
            return
        tx.mutate(f"DELETE FROM {model._table} WHERE id=?", (model.id,))
        tx.mutate(
            f"DELETE FROM {model._flex_table} WHERE entity_id=?", (model.id,)
        )

    @staticmethod
    def _restore_row(tx: Any, model: Model) -> None:
        """Re-insert a snapshotted row with its original id and fields.

        Rows that still exist (e.g. an album that was never actually
        removed) are left untouched.
        """
        if model.id is None:
            return
        if tx.query(f"SELECT 1 FROM {model._table} WHERE id=?", (model.id,)):
            return
        fields = [k for k in model._fields if k in model._values_fixed]
        columns = ",".join(fields)
        placeholders = ",".join("?" * len(fields))
        tx.mutate(
            f"INSERT INTO {model._table} ({columns}) VALUES ({placeholders})",
            [
                model._type(key).to_sql(model._values_fixed[key])
                for key in fields
            ],
        )
        for key, value in model._values_flex.items():
            tx.mutate(
                f"INSERT INTO {model._flex_table} "
                "(entity_id, key, value) VALUES (?, ?, ?)",
                (model.id, key, model._type(key).to_sql(value)),
            )
