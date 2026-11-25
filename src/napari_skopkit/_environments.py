"""The Environments panel: which op environments exist, and building them.

Each op runs in an environment of its own, built on first use -- minutes and
gigabytes, at whatever moment someone first presses Run. This panel shows
each one's state ahead of time, and builds on demand:

* **Ready** -- built from the recipe now installed.
* **Needs rebuild** -- built from an older recipe. The next use rebuilds it,
  as a new release with a changed recipe requires.
* **Not built** -- the next use builds it from scratch.

Listed are the environments the installed op collections bring (every
``opspec.collections`` entry point, as the Ops panel finds them): one row
each, however many ops share it. Anything else in appose's directory -- old
folders, Fiji's -- is under "Show other environments on disk", with a filter,
and can be deleted from here. "Clean download cache" deletes what pixi's
package cache holds that no environment uses any more, e.g. the torch a
rebuild replaced (see ``_cache``).

Everything it knows comes from oprunner's public API: ``collection_envs``,
``Env.status``, ``Env.log``, ``Runner.build`` and the build subscriptions.

An application that runs ops through a runner of its own -- napari-ai-lab --
subclasses this and overrides the hooks under "Hooks" below, so that builds
its segmenters start show here, and "Used by" names its segmenters rather
than ops.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from pathlib import Path

from qtpy.QtCore import QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import ensure_main_thread

from . import _cache

READY, STALE, MISSING, OTHER = "Ready", "Needs rebuild", "Not built", "On disk"
BUILDING, FAILED = "Building…", "Failed"
_STATUS = {"ready": READY, "stale": STALE, "missing": MISSING}

#: "[environment build] Installing conda packages 12/40": the line format the
#: default listen() produces, and napari-ai-lab's build reporter too.
_PROGRESS = re.compile(r"\[environment build\] (.*?)(?: (\d+)/(\d+))?$")

#: How many op names "Used by" shows before it counts the rest.
_SHOWN_USERS = 4


def _appose_root() -> Path | None:
    try:
        from appose.util.filepath import appose_envs_dir
    except ImportError:
        try:
            from appose.util.environment import appose_envs_dir
        except ImportError:
            return None
    return Path(appose_envs_dir())


def _size(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, f)).st_size
    return total


def _human(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _shorten(names: list[str]) -> str:
    if len(names) <= _SHOWN_USERS:
        return ", ".join(names)
    return f"{', '.join(names[:_SHOWN_USERS])} (+{len(names) - _SHOWN_USERS} more)"


# -- threads ----------------------------------------------------------------


class _BuildThread(QThread):
    """Builds one environment off the GUI thread."""

    done = Signal(str, bool, str)  # env name, ok, message

    def __init__(self, runner, env):
        super().__init__()
        self.runner = runner
        self.env = env

    def run(self):
        try:
            self.runner.build(self.env)
            self.done.emit(self.env.name, True, "")
        except Exception as e:  # noqa: BLE001 - reported in the log
            self.done.emit(self.env.name, False, str(e))


class _SizeThread(QThread):
    """Adds up folder sizes in the background: a torch stack is 100k files."""

    sized = Signal(str, str)  # env name, size text

    def __init__(self, folders: dict[str, Path]):
        super().__init__()
        self.folders = folders

    def run(self):
        for name, folder in self.folders.items():
            if self.isInterruptionRequested():
                return
            if folder.is_dir():
                self.sized.emit(name, _human(_size(folder)))


class _PruneScanThread(QThread):
    """Finds the cache entries no environment uses: reads every file's links."""

    found = Signal(object)  # list[_cache.Entry]

    def __init__(self, pkgs: Path):
        super().__init__()
        self.pkgs = pkgs

    def run(self):
        entries = _cache.find_prunable(self.pkgs, self.isInterruptionRequested)
        self.found.emit(entries)


class _PruneThread(QThread):
    """Deletes cache entries."""

    done = Signal(int, object)  # bytes freed, list of error strings

    def __init__(self, entries):
        super().__init__()
        self.entries = entries

    def run(self):
        freed, errors = _cache.delete(self.entries)
        self.done.emit(freed, errors)


# -- the panel --------------------------------------------------------------


class EnvironmentsPanel(QWidget):
    """Status of every op environment, and a button to build each."""

    COLUMNS = ("Environment", "Used by", "Status", "Size")

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._envs: dict[str, object] = {}  # name -> Env, or None if "other"
        self._failed: dict[str, str] = {}
        self._building: dict[str, _BuildThread] = {}
        self._size_thread = None
        self._prune_thread = None
        self._setup_ui()
        self.listen(ensure_main_thread(self._on_build_news))
        self.refresh()

    # Hooks ------------------------------------------------------------------
    # What an application running its own runner overrides.

    def runner(self):
        """The runner whose builds this panel starts and hears."""
        from oprunner.runner import default_runner

        return default_runner()

    def collections(self) -> dict[str, str]:
        """Collection name -> the package of its ops, e.g. "skop.ops"."""
        from ._panel import _collections

        return _collections()

    def users(self) -> dict[str, list[str]]:
        """Environment name -> what runs in it, for the "Used by" column.

        The default lists each collection's ops by name.
        """
        from oprunner import discover

        runner = self.runner()
        found: dict[str, list[str]] = {}
        for package in self.collections().values():
            names = {e.id: e.name for e in runner.collection_envs(package)}
            specs, _failures = discover(package)
            for spec in specs:
                if spec.env in names:
                    found.setdefault(names[spec.env], []).append(
                        spec.name.split(":")[-1]
                    )
        return found

    def listen(self, on_line) -> None:
        """Have *on_line* hear every build's news, one line of text at a time.

        The default subscribes to this panel's runner. ``on_line`` already
        hops to the GUI thread, so it may be called from any thread.
        """

        def progress(title, current, maximum):
            counts = f" {current}/{maximum}" if maximum else ""
            on_line(f"[environment build] {title}{counts}")

        self.runner().subscribe_build_progress(progress)

    def say(self, line: str) -> None:
        """Announce *line*, e.g. "Building ...". The default logs it here."""
        self._on_build_news(line)

    def release_workers(self) -> None:
        """Stop the workers that may hold an environment's files open, before
        a delete."""
        self.runner().close()

    # UI ---------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each op runs in an environment of its own, built on first "
            "use: minutes and gigabytes, once. Build them here ahead of "
            "time, rebuild one a new release has changed, or delete old ones."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table, 1)

        others = QHBoxLayout()
        self.show_other = QCheckBox("Show other environments on disk")
        self.show_other.toggled.connect(self._apply_filter)
        others.addWidget(self.show_other)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter by name")
        self.filter_edit.textChanged.connect(self._apply_filter)
        others.addWidget(self.filter_edit, 1)
        layout.addLayout(others)

        buttons = QHBoxLayout()
        self.build_btn = QPushButton("Build")
        self.build_btn.clicked.connect(self._on_build)
        self.log_btn = QPushButton("Show log")
        self.log_btn.clicked.connect(self._on_show_log)
        self.folder_btn = QPushButton("Open folder")
        self.folder_btn.clicked.connect(self._on_open_folder)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        for b in (
            self.build_btn,
            self.log_btn,
            self.folder_btn,
            self.delete_btn,
            refresh_btn,
        ):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.prune_btn = QPushButton("Clean download cache")
        self.prune_btn.setToolTip(
            "Delete the packages pixi's download cache holds that no "
            "environment uses any more -- e.g. the torch a rebuild replaced. "
            "Environments keep working."
        )
        self.prune_btn.clicked.connect(self._on_prune)
        layout.addWidget(self.prune_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.log_view, 1)

    def _log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # Contents ---------------------------------------------------------------

    def refresh(self):
        """Re-read every environment's state from disk."""
        try:
            runner = self.runner()
        except Exception as e:  # noqa: BLE001 - scikit-ops missing
            self._log(f"No op runner available: {e}")
            return
        try:
            users = self.users()
        except Exception:  # noqa: BLE001 - informational only
            users = {}

        self._envs = {}
        for package in self.collections().values():
            for env in runner.collection_envs(package):
                self._envs.setdefault(env.name, env)
        root = _appose_root()
        if root is not None and root.is_dir():
            for folder in sorted(root.iterdir()):
                if folder.is_dir() and not folder.name.startswith("."):
                    self._envs.setdefault(folder.name, None)

        # Ours first, the ones in use first among them; then the rest.
        def order(name):
            env = self._envs[name]
            return (env is None, name not in users, name)

        names = sorted(self._envs, key=order)
        selected = self._selected_name()
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            env = self._envs[name]
            used = users.get(name, [])
            self.table.setItem(row, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(_shorten(used))
            if used:
                item.setToolTip("\n".join(used))
            self.table.setItem(row, 1, item)
            self.table.setItem(row, 2, QTableWidgetItem(self._status_of(name)))
            self.table.setItem(row, 3, QTableWidgetItem(""))
            if env is not None:
                self.table.item(row, 0).setToolTip(str(env.config))
            if name == selected:
                self.table.selectRow(row)
        self._apply_filter()
        self._update_buttons()
        self._start_sizes()

    def _status_of(self, name) -> str:
        if name in self._building:
            return BUILDING
        env = self._envs.get(name)
        if env is None:
            return OTHER
        if name in self._failed:
            return FAILED
        return _STATUS.get(env.status, MISSING)

    def _start_sizes(self):
        if self._size_thread is not None:
            self._size_thread.requestInterruption()
            self._size_thread.wait(2000)
        root = _appose_root()
        if root is None:
            return
        self._size_thread = _SizeThread({n: root / n for n in self._envs})
        self._size_thread.sized.connect(self._on_sized)
        self._size_thread.start()

    def _on_sized(self, name, text):
        row = self._row_of(name)
        if row is not None:
            self.table.item(row, 3).setText(text)

    def _apply_filter(self, *_):
        text = self.filter_edit.text().strip().lower()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            other = self._envs.get(name) is None
            hidden = (other and not self.show_other.isChecked()) or (
                text and text not in name.lower()
            )
            self.table.setRowHidden(row, bool(hidden))

    def _row_of(self, name):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == name:
                return row
        return None

    def _selected_name(self):
        rows = self.table.selectionModel().selectedRows()
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def _set_status(self, name, status):
        row = self._row_of(name)
        if row is not None:
            self.table.item(row, 2).setText(status)

    def _update_buttons(self):
        name = self._selected_name()
        env = self._envs.get(name) if name else None
        status = self._status_of(name) if name else None
        self.build_btn.setEnabled(env is not None and status != BUILDING)
        self.build_btn.setText(
            "Rebuild" if status in (READY, STALE, FAILED) else "Build"
        )
        self.log_btn.setEnabled(env is not None)
        root = _appose_root()
        on_disk = name is not None and root is not None and (root / name).is_dir()
        self.folder_btn.setEnabled(on_disk)
        self.delete_btn.setEnabled(on_disk and status != BUILDING)

    # Actions ----------------------------------------------------------------

    def _on_build(self):
        name = self._selected_name()
        env = self._envs.get(name)
        if env is None or name in self._building:
            return
        if env.status == "ready" and not self._confirm(
            "Rebuild",
            f"'{name}' is already built from the current recipe. "
            "Rebuilding re-runs pixi, which installs nothing new. Go ahead?",
        ):
            return
        self._failed.pop(name, None)
        self.say(f"Building the '{name}' environment, from {env.config}")
        thread = _BuildThread(self.runner(), env)
        thread.done.connect(self._on_build_done)
        self._building[name] = thread
        self._set_status(name, BUILDING)
        self._update_buttons()
        thread.start()

    def _on_build_done(self, name, ok, message):
        thread = self._building.pop(name, None)
        if thread is not None:
            thread.wait()
        if ok:
            self.say(f"✅ The '{name}' environment is built and ready")
            self.progress_bar.setValue(100)
        else:
            self._failed[name] = message
            # oprunner's BuildFailed names the environment and the log.
            self.say(f"❌ {message}")
        self.refresh()

    def _on_build_news(self, line):
        """A line from any build: log it, and move the bar if it says how far."""
        self._log(line)
        found = _PROGRESS.match(line.strip())
        if found and found.group(3):
            current, total = int(found.group(2)), int(found.group(3))
            if total:
                self.progress_bar.setValue(int(100 * current / total))

    def _on_show_log(self):
        env = self._envs.get(self._selected_name())
        if env is None:
            return
        if env.log.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(env.log)))
        else:
            self._log(f"No build log for '{env.name}' yet.")

    def _on_open_folder(self):
        name = self._selected_name()
        root = _appose_root()
        if name and root is not None and (root / name).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root / name)))

    def _on_delete(self):
        name = self._selected_name()
        root = _appose_root()
        if not name or root is None or name in self._building:
            return
        folder = root / name
        if not folder.is_dir() or not self._confirm(
            "Delete environment",
            f"Delete {folder}?\n\nThe next use builds it again from scratch, "
            "which takes minutes and downloads gigabytes. Running models are "
            "stopped first.",
        ):
            return
        self.release_workers()  # a running worker holds its environment's files
        try:
            shutil.rmtree(folder)
            self._log(f"Deleted {folder}")
        except OSError as e:
            self._log(f"Could not delete {folder}: {e}")
        self.refresh()

    def _on_prune(self):
        if self._building:
            self._log("Wait for the build to finish: it is using the cache.")
            return
        pkgs = _cache.conda_pkgs_dir()
        if pkgs is None:
            self._log("No pixi package cache found.")
            return
        self._log(f"Looking for unused packages in {pkgs} ...")
        self.prune_btn.setEnabled(False)
        self._prune_thread = _PruneScanThread(pkgs)
        self._prune_thread.found.connect(self._on_prune_found)
        self._prune_thread.start()

    def _on_prune_found(self, entries):
        self._prune_thread = None
        self.prune_btn.setEnabled(True)
        if not entries:
            self._log("Nothing to clean: every cached package is in use.")
            return
        unused = [e for e in entries if e.reason == _cache.UNUSED]
        unfinished = [e for e in entries if e.reason == _cache.UNFINISHED]
        total = sum(e.size for e in entries)
        biggest = sorted(entries, key=lambda e: e.size, reverse=True)[:8]
        lines = "\n".join(f"  {_human(e.size):>8}  {e.path.name}" for e in biggest)
        text = (
            f"Delete {len(unused)} unused and {len(unfinished)} unfinished "
            f"packages, {_human(total)}, from\n{entries[0].path.parent}?\n\n"
            f"Largest:\n{lines}\n\n"
            "Environments keep working. A later build downloads again what "
            "it needs. Close other pixi builds first."
        )
        if not self._confirm("Clean download cache", text):
            return
        self.prune_btn.setEnabled(False)
        self._prune_thread = _PruneThread(entries)
        self._prune_thread.done.connect(self._on_pruned)
        self._prune_thread.start()

    def _on_pruned(self, freed, errors):
        self._prune_thread = None
        self.prune_btn.setEnabled(True)
        self._log(f"Cleaned the download cache: {_human(freed)} freed.")
        for error in errors[:10]:
            self._log(f"Could not delete {error}")
        if len(errors) > 10:
            self._log(f"... and {len(errors) - 10} more")

    def _confirm(self, title, text) -> bool:
        answer = QMessageBox.question(
            self, title, text, QMessageBox.Yes | QMessageBox.No
        )
        return answer == QMessageBox.Yes

    def closeEvent(self, event):
        if self._size_thread is not None:
            self._size_thread.requestInterruption()
        if isinstance(self._prune_thread, _PruneScanThread):
            self._prune_thread.requestInterruption()
        super().closeEvent(event)
