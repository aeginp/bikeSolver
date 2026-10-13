"""PySide6 window for editing a frame project and reading member results."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bike_fea.frame import MEMBER_CATALOG, Analysis, FrameProject, MemberSize, analyze
from bike_fea.geometry import FrameGeometry, GeometryError, build_geometry
from bike_fea.loads import LoadCase
from bike_fea.materials import BUILTINS, Material
from bike_fea.plotting import draw_frame
from bike_fea.project import EXAMPLE_PATH, example_project, load_project, save_project

FRAME_FIELDS: tuple[tuple[str, str, float, float, int], ...] = (
    ("wheel_diameter", "wheel_diameter (mm)", 0, 1000, 1),
    ("axle_to_crown", "axle_to_crown (mm)", 0, 800, 1),
    ("fork_offset", "fork_offset (mm)", 0, 150, 1),
    ("head_tube_angle", "head_tube_angle (deg)", 1, 179, 2),
    ("head_tube", "head_tube (mm)", 0, 500, 1),
    ("bb_drop", "bb_drop (mm)", 0, 500, 1),
    ("seat_tube_angle", "seat_tube_angle (deg)", 1, 179, 2),
    ("seat_tube", "seat_tube, c-c (mm)", 0, 1500, 1),
    ("top_tube", "top_tube, c-c (mm)", 0, 1500, 1),
    ("chainstay", "chainstay (mm)", 0, 1000, 1),
    ("saddle_height", "saddle_height (mm)", 0, 1500, 1),
)

DETAIL_FIELDS: tuple[tuple[str, str, float, float, int], ...] = (
    ("seatstay_drop", "seatstay_drop (mm)", 0, 800, 1),
    ("seat_tube_extension", "seat_tube_extension (mm)", 0, 500, 1),
    ("stem_length", "stem_length (mm)", 0, 300, 1),
    ("stem_angle", "stem_angle, +up (deg)", -80, 80, 1),
    ("spacers", "spacers (mm)", 0, 80, 1),
)

GEOMETRY_FIELDS = FRAME_FIELDS + DETAIL_FIELDS

DERIVED_FIELDS: tuple[tuple[str, str], ...] = (
    ("stack", "Stack"),
    ("reach", "Reach"),
    ("effective_top_tube", "Effective top tube"),
    ("rear_center", "Rear center"),
    ("front_center", "Front center"),
    ("wheelbase", "Wheelbase"),
    ("trail", "Trail"),
    ("bb_height", "BB height"),
)

RESULT_COLUMNS = (
    "Member",
    "Case",
    "Axial (N)",
    "Sense",
    "Peak moment (N·mm)",
    "Peak stress (MPa)",
    "At",
    "Safety factor",
    "Required wall (mm)",
    "Leg axial (N)",
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bike frame FEA")
        self.resize(1400, 860)
        self._path: Path | None = None
        self._filling = False
        self._last_geometry = None
        self._last_analysis: Analysis | None = None
        self._project = example_project()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._resolve)

        self._geometry_spins: dict[str, QDoubleSpinBox] = {}
        self._build_ui()
        self._show_project(self._project, None)

    def _build_ui(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("New", self._new)
        file_menu.addAction("Open…", self._open)
        file_menu.addAction("Save", self._save)
        file_menu.addSeparator()
        file_menu.addAction("Export CSV…", self._export_csv)
        file_menu.addAction("Export PNG…", self._export_png)

        tabs = QTabWidget()
        tabs.addTab(self._geometry_tab(), "Geometry")
        tabs.addTab(self._members_tab(), "Members")
        tabs.addTab(self._material_tab(), "Material")
        tabs.addTab(self._loads_tab(), "Load cases")
        tabs.addTab(self._settings_tab(), "Settings")
        tabs.setMinimumWidth(400)

        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._color_combo = QComboBox()
        self._color_combo.addItems(["Stress", "Axial", "Safety factor"])
        self._case_combo = QComboBox()
        self._case_combo.addItem("Worst case")
        self._deformed = QCheckBox("Deformed shape")
        self._deformed.setChecked(True)
        self._color_combo.currentIndexChanged.connect(self._redraw)
        self._case_combo.currentIndexChanged.connect(self._redraw)
        self._deformed.toggled.connect(self._redraw)

        self._solve_button = QPushButton("Solve")
        self._solve_button.clicked.connect(self._solve_now)
        self._auto = QCheckBox("Automatic compute")
        self._auto.setChecked(True)
        self._auto.toggled.connect(self._on_auto_toggled)
        self._stale = QLabel("")
        self._stale.setStyleSheet("color: #c0392b;")

        tools = QHBoxLayout()
        tools.addWidget(self._solve_button)
        tools.addWidget(self._auto)
        tools.addWidget(self._stale)
        tools.addSpacing(16)
        tools.addWidget(QLabel("Color"))
        tools.addWidget(self._color_combo)
        tools.addWidget(QLabel("Case"))
        tools.addWidget(self._case_combo)
        tools.addWidget(self._deformed)
        tools.addStretch(1)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addLayout(tools)
        center_layout.addWidget(self.canvas, stretch=1)

        self._results = QTableWidget(0, len(RESULT_COLUMNS))
        self._results.setHorizontalHeaderLabels(list(RESULT_COLUMNS))
        self._results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results.verticalHeader().setVisible(False)
        self._results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._results.setMinimumWidth(420)

        self._status = QPlainTextEdit()
        self._status.setReadOnly(True)
        self._status.setFixedHeight(110)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tabs)
        splitter.addWidget(center)
        splitter.addWidget(self._results)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 560, 420])

        outer = QVBoxLayout()
        outer.addWidget(splitter, stretch=1)
        outer.addWidget(self._status)
        holder = QWidget()
        holder.setLayout(outer)
        self.setCentralWidget(holder)

    def _geometry_tab(self) -> QWidget:
        layout = QVBoxLayout()
        for title, group_fields in (("Frame", FRAME_FIELDS), ("Cockpit and details", DETAIL_FIELDS)):
            form = QFormLayout()
            for key, label, low, high, decimals in group_fields:
                spin = QDoubleSpinBox()
                spin.setRange(low, high)
                spin.setDecimals(decimals)
                spin.setSingleStep(1 if decimals == 1 else 0.1)
                spin.valueChanged.connect(self._schedule)
                self._geometry_spins[key] = spin
                form.addRow(label, spin)
            box = QGroupBox(title)
            box.setLayout(form)
            layout.addWidget(box)

        derived_form = QFormLayout()
        self._derived_labels: dict[str, QLabel] = {}
        for key, label in DERIVED_FIELDS:
            value = QLabel("–")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._derived_labels[key] = value
            derived_form.addRow(label, value)
        derived_box = QGroupBox("Derived (read-only)")
        derived_box.setLayout(derived_form)
        layout.addWidget(derived_box)
        layout.addStretch(1)

        holder = QWidget()
        holder.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        return scroll

    def _members_tab(self) -> QWidget:
        self._members = QTableWidget(len(MEMBER_CATALOG), 2)
        self._members.setHorizontalHeaderLabels(["OD (mm)", "Wall (mm)"])
        self._members.setVerticalHeaderLabels([name for _key, name, _pair in MEMBER_CATALOG])
        self._members.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._members.itemChanged.connect(self._schedule)
        note = QLabel("Outside diameter and wall. Stays and fork are one member for the pair.")
        note.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(note)
        layout.addWidget(self._members)
        holder = QWidget()
        holder.setLayout(layout)
        return holder

    def _material_tab(self) -> QWidget:
        self._material_combo = QComboBox()
        self._material_combo.addItems([*BUILTINS.keys(), "Custom"])
        self._material_combo.currentIndexChanged.connect(self._on_material_changed)
        self._material_name = QLineEdit()
        self._material_name.textChanged.connect(self._schedule)
        self._modulus = self._spin(1000, 500_000, 0, 55000)
        self._yield = self._spin(1, 5000, 1, 500)
        self._compression = self._spin(0, 5000, 1, 0)
        form = QFormLayout()
        form.addRow("Material", self._material_combo)
        form.addRow("Name", self._material_name)
        form.addRow("E (MPa)", self._modulus)
        form.addRow("Yield (MPa)", self._yield)
        form.addRow("Compression allowable (MPa)", self._compression)
        note = QLabel("Leave compression allowable at 0 to use yield for both signs. Custom is how an isotropic carbon estimate is entered.")
        note.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addStretch(1)
        holder = QWidget()
        holder.setLayout(layout)
        return holder

    def _loads_tab(self) -> QWidget:
        self._loads = QTableWidget(0, 6)
        self._loads.setHorizontalHeaderLabels(["Name", "Mass (kg)", "Bump g", "Saddle %", "Bars %", "Pedals %"])
        self._loads.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._loads.itemChanged.connect(self._schedule)
        add = QPushButton("Add case")
        remove = QPushButton("Remove case")
        add.clicked.connect(self._add_case)
        remove.clicked.connect(self._remove_case)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        note = QLabel("Bump factor is total load. 1 is sitting still. The three percentages must sum to 100.")
        note.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(note)
        layout.addWidget(self._loads)
        layout.addLayout(buttons)
        holder = QWidget()
        holder.setLayout(layout)
        return holder

    def _settings_tab(self) -> QWidget:
        self._target = self._spin(0.5, 20, 2, 2)
        form = QFormLayout()
        form.addRow("Target safety factor", self._target)
        note = QLabel("Required wall is the thinnest wall, at the current outside diameter, that meets this factor.")
        note.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addStretch(1)
        holder = QWidget()
        holder.setLayout(layout)
        return holder

    def _spin(self, low: float, high: float, decimals: int, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.valueChanged.connect(self._schedule)
        return spin

    def _show_project(self, project: FrameProject, path: Path | None) -> None:
        self._filling = True
        self._path = path
        self._project = project
        try:
            for key, _label, _low, _high, _decimals in GEOMETRY_FIELDS:
                self._geometry_spins[key].setValue(getattr(project.geometry, key))
            for row, (key, _name, _pair) in enumerate(MEMBER_CATALOG):
                size = project.members[key]
                self._set_cell(self._members, row, 0, size.od)
                self._set_cell(self._members, row, 1, size.wall)
            if project.material.name in BUILTINS and project.material == BUILTINS[project.material.name]:
                self._material_combo.setCurrentText(project.material.name)
            else:
                self._material_combo.setCurrentText("Custom")
                self._material_name.setText(project.material.name)
                self._modulus.setValue(project.material.youngs_modulus)
                self._yield.setValue(project.material.yield_strength)
                compression = project.material.allowable_compression
                self._compression.setValue(0 if compression is None else compression)
            self._set_custom_enabled(self._material_combo.currentText() == "Custom")
            self._loads.setRowCount(0)
            for case in project.load_cases:
                self._append_case(case.name, case.mass_kg, case.bump_g, case.saddle * 100, case.bars * 100, case.pedals * 100)
            self._target.setValue(project.target_safety_factor)
        finally:
            self._filling = False
        self._title()
        self._resolve()

    def _schedule(self) -> None:
        if self._filling:
            return
        if self._auto.isChecked():
            self._timer.start()
        else:
            self._stale.setText("Inputs changed. Press Solve.")

    def _solve_now(self) -> None:
        self._timer.stop()
        self._resolve()

    def _on_auto_toggled(self, checked: bool) -> None:
        if checked and self._stale.text():
            self._solve_now()

    def _on_material_changed(self) -> None:
        custom = self._material_combo.currentText() == "Custom"
        self._set_custom_enabled(custom)
        if custom and not self._filling and not self._material_name.text().strip():
            self._material_name.setText("custom")
            self._modulus.setValue(205_000)
            self._yield.setValue(460)
        self._schedule()

    def _set_custom_enabled(self, enabled: bool) -> None:
        for widget in (self._material_name, self._modulus, self._yield, self._compression):
            widget.setEnabled(enabled)

    def _add_case(self) -> None:
        self._append_case("case", 75, 1, 60, 30, 10)
        self._schedule()

    def _append_case(self, name: str, mass: float, bump: float, saddle: float, bars: float, pedals: float) -> None:
        row = self._loads.rowCount()
        self._loads.insertRow(row)
        values = (name, mass, bump, saddle, bars, pedals)
        for column, value in enumerate(values):
            text = value if isinstance(value, str) else f"{value:g}"
            self._loads.setItem(row, column, QTableWidgetItem(text))

    def _remove_case(self) -> None:
        row = self._loads.currentRow()
        if row < 0 and self._loads.rowCount():
            row = self._loads.rowCount() - 1
        if row >= 0:
            self._loads.removeRow(row)
            self._schedule()

    def _resolve(self) -> None:
        self._stale.setText("")
        try:
            project = self._read_project()
        except ValueError as exc:
            self._fail_geometry(str(exc))
            return
        self._project = project
        try:
            geometry = build_geometry(project.geometry)
        except GeometryError as exc:
            self._fail_geometry(str(exc))
            return
        try:
            analysis = analyze(project)
        except ValueError as exc:
            self._last_geometry = geometry
            self._last_analysis = None
            self._draw(geometry, None, str(exc))
            self._fill_results(None)
            self._fill_status(geometry, None, str(exc))
            return
        self._last_geometry = geometry
        self._last_analysis = analysis
        self._sync_cases(analysis)
        self._draw(geometry, analysis, None)
        self._fill_results(analysis)
        self._fill_status(geometry, analysis, None)

    def _redraw(self) -> None:
        warning = None
        if self._last_geometry is None:
            warning = self._status.toPlainText() or None
        self._draw(self._last_geometry, self._last_analysis, warning if self._last_analysis is None else None)
        self._fill_results(self._last_analysis)
        if self._last_geometry is not None:
            self._fill_status(self._last_geometry, self._last_analysis, None)

    def _draw(self, geometry, analysis, warning: str | None) -> None:
        if geometry is None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, warning or "No geometry yet.", ha="center", va="center", color="tab:red", wrap=True)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        draw_frame(
            self.figure,
            geometry,
            analysis,
            color_by=self._color_key(),
            case_name=self._case_name(),
            deformed=self._deformed.isChecked(),
            load_cases=self._project.load_cases,
            warning=warning,
        )
        self.canvas.draw_idle()

    def _color_key(self) -> str:
        return {"Stress": "stress", "Axial": "axial", "Safety factor": "sf"}[self._color_combo.currentText()]

    def _case_name(self) -> str | None:
        text = self._case_combo.currentText()
        if not text or text == "Worst case":
            return None
        return text

    def _sync_cases(self, analysis: Analysis) -> None:
        current = self._case_combo.currentText() or "Worst case"
        self._case_combo.blockSignals(True)
        self._case_combo.clear()
        self._case_combo.addItem("Worst case")
        for case in analysis.cases:
            self._case_combo.addItem(case.name)
        index = self._case_combo.findText(current)
        self._case_combo.setCurrentIndex(max(index, 0))
        self._case_combo.blockSignals(False)

    def _fill_results(self, analysis: Analysis | None) -> None:
        self._results.setRowCount(0)
        if analysis is None:
            return
        case_name = self._case_name()
        walls = {item.force.key: item.required_wall for item in analysis.summaries}
        if case_name is None:
            rows = [(item.case_name, item.force, item.required_wall) for item in analysis.summaries]
        else:
            case = next(item for item in analysis.cases if item.name == case_name)
            rows = [(case.name, member, walls.get(member.key, float("nan"))) for member in case.members]
        self._results.setRowCount(len(rows))
        for row_index, (name, force, wall) in enumerate(rows):
            peak_moment = force.moment_start if abs(force.moment_start) >= abs(force.moment_end) else force.moment_end
            leg = "" if force.axial_per_leg is None else f"{force.axial_per_leg:.1f}"
            values = [
                force.display,
                name,
                f"{force.axial:.1f}",
                force.sense,
                f"{peak_moment:.0f}",
                f"{force.stress:.1f}",
                force.peak_at,
                _format_factor(force.safety_factor),
                f"{wall:.2f}",
                leg,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._results.setItem(row_index, column, item)

    def _fill_status(self, geometry, analysis: Analysis | None, warning: str | None) -> None:
        derived = geometry.derived
        for key, label in self._derived_labels.items():
            label.setText(f"{getattr(derived, key):.1f} mm")
        lines = [
            f"stack {derived.stack:.1f} mm    reach {derived.reach:.1f} mm    "
            f"wheelbase {derived.wheelbase:.1f} mm    trail {derived.trail:.1f} mm"
        ]
        if analysis is not None:
            selected = self._case_name()
            cases = analysis.cases if selected is None else [case for case in analysis.cases if case.name == selected]
            for case in cases:
                lines.append(
                    f"{case.name}: rear {case.rear_fy:.0f} N up, horizontal {case.rear_fx:+.1f} N, "
                    f"front {case.front_fy:.0f} N up, residual {case.residual_fy:.1e} N"
                )
                lines.append(
                    f"  BB {case.bb_deflection:.2f} mm    bars vs front axle {case.bar_deflection:.2f} mm"
                )
        if warning:
            lines.append(warning)
        self._status.setPlainText("\n".join(lines))

    def _fail_geometry(self, message: str) -> None:
        self._draw(self._last_geometry, self._last_analysis, message)
        if self._last_geometry is None:
            self._status.setPlainText(message)
        else:
            self._fill_status(self._last_geometry, self._last_analysis, message)

    def _read_project(self) -> FrameProject:
        geometry = FrameGeometry(**{key: spin.value() for key, spin in self._geometry_spins.items()})
        members: dict[str, MemberSize] = {}
        for row, (key, name, _pair) in enumerate(MEMBER_CATALOG):
            members[key] = MemberSize(self._cell_float(self._members, row, 0, name, "OD"), self._cell_float(self._members, row, 1, name, "wall"))
        if self._material_combo.currentText() == "Custom":
            name = self._material_name.text().strip()
            if not name:
                raise ValueError("Custom material needs a name.")
            compression = self._compression.value()
            material = Material(
                name,
                self._modulus.value(),
                self._yield.value(),
                None if compression == 0 else compression,
            )
        else:
            material = BUILTINS[self._material_combo.currentText()]
        cases = []
        for row in range(self._loads.rowCount()):
            label = self._cell_text(self._loads, row, 0) or f"case {row + 1}"
            cases.append(
                LoadCase(
                    label,
                    self._cell_float(self._loads, row, 1, label, "mass"),
                    self._cell_float(self._loads, row, 2, label, "bump factor"),
                    self._cell_float(self._loads, row, 3, label, "saddle") / 100.0,
                    self._cell_float(self._loads, row, 4, label, "bars") / 100.0,
                    self._cell_float(self._loads, row, 5, label, "pedals") / 100.0,
                )
            )
        return FrameProject(geometry, material, members, tuple(cases), self._target.value())

    def _cell_text(self, table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return "" if item is None else item.text().strip()

    def _cell_float(self, table: QTableWidget, row: int, column: int, owner: str, field: str) -> float:
        text = self._cell_text(table, row, column)
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{owner} {field} is not a number.") from exc

    def _set_cell(self, table: QTableWidget, row: int, column: int, value: float) -> None:
        table.setItem(row, column, QTableWidgetItem(f"{value:g}"))

    def _new(self) -> None:
        self._show_project(example_project(), None)

    def _open(self) -> None:
        start = str(self._path.parent if self._path else EXAMPLE_PATH.parent)
        path, _selected = QFileDialog.getOpenFileName(self, "Open project", start, "TOML (*.toml)")
        if path:
            self._show_project(load_project(path), Path(path))

    def _save(self) -> None:
        if self._path is None:
            start = str(EXAMPLE_PATH.parent)
            path, _selected = QFileDialog.getSaveFileName(self, "Save project", start, "TOML (*.toml)")
            if not path:
                return
            self._path = Path(path)
        try:
            project = self._read_project()
        except ValueError as exc:
            self._fail_geometry(str(exc))
            return
        save_project(project, self._path)
        self._project = project
        self._title()

    def _export_csv(self) -> None:
        if self._last_analysis is None:
            return
        path, _selected = QFileDialog.getSaveFileName(self, "Export CSV", "results.csv", "CSV (*.csv)")
        if path:
            write_results_csv(self._last_analysis, Path(path), self._case_name())

    def _export_png(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(self, "Export PNG", "frame.png", "PNG (*.png)")
        if path:
            self.figure.savefig(path, dpi=150)

    def _title(self) -> None:
        name = self._path.name if self._path else "untitled"
        self.setWindowTitle(f"Bike frame FEA — {name}")


def write_results_csv(analysis: Analysis, path: Path, case_name: str | None = None) -> None:
    walls = {item.force.key: item.required_wall for item in analysis.summaries}
    if case_name is None:
        rows = [(item.case_name, item.force, item.required_wall) for item in analysis.summaries]
    else:
        case = next(item for item in analysis.cases if item.name == case_name)
        rows = [(case.name, member, walls.get(member.key, float("nan"))) for member in case.members]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "member",
                "case",
                "axial_n",
                "sense",
                "peak_moment_nmm",
                "peak_stress_mpa",
                "peak_at",
                "safety_factor",
                "required_wall_mm",
                "axial_per_leg_n",
            ]
        )
        for name, force, wall in rows:
            peak = force.moment_start if abs(force.moment_start) >= abs(force.moment_end) else force.moment_end
            writer.writerow(
                [
                    force.display,
                    name,
                    f"{force.axial:.2f}",
                    force.sense,
                    f"{peak:.1f}",
                    f"{force.stress:.2f}",
                    force.peak_at,
                    _format_factor(force.safety_factor),
                    f"{wall:.2f}",
                    "" if force.axial_per_leg is None else f"{force.axial_per_leg:.2f}",
                ]
            )


def _format_factor(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
