import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from bike_fea.app import MainWindow


def test_window_opens_on_a_solved_example():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window._last_analysis is not None
        assert window._results.rowCount() == 10
        assert "ETT" in window._status.toPlainText()
        window._geometry_spins["chainstay"].setValue(10)
        window._resolve()
        assert window._last_analysis is not None
        assert "shorter" in window._status.toPlainText().lower()
    finally:
        window.close()
        app.processEvents()
