import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from bike_fea.app import MainWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    main = MainWindow()
    yield main
    main.close()
    app.processEvents()


def test_window_opens_on_a_solved_example(window):
    assert window._last_analysis is not None
    assert window._results.rowCount() == 8
    assert window._derived_labels["front_center"].text() == "615.7 mm"
    window._geometry_spins["chainstay"].setValue(10)
    window._solve_now()
    assert window._last_analysis is not None
    assert "shorter" in window._status.toPlainText().lower()


def test_manual_solve_waits_for_the_button(window):
    window._auto.setChecked(False)
    before = window._last_geometry.derived.front_center
    window._geometry_spins["top_tube"].setValue(560)
    assert window._stale.text()
    assert window._last_geometry.derived.front_center == before
    window._solve_button.click()
    assert window._stale.text() == ""
    assert window._last_geometry.derived.front_center > before
