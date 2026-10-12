from matplotlib.figure import Figure

from bike_fea.frame import analyze
from bike_fea.plotting import draw_frame
from bike_fea.project import example_project


def test_plot_colors_members_and_labels_nodes():
    project = example_project()
    analysis = analyze(project)
    figure = Figure()
    draw_frame(figure, analysis.geometry, analysis, color_by="stress", deformed=True, load_cases=project.load_cases)
    ax = figure.axes[0]
    labels = [text.get_text() for text in ax.texts]
    assert "BB" in labels
    assert "stem" in labels
    assert ax.collections
    assert "Worst case" in ax.get_title()


def test_invalid_geometry_is_not_required_to_draw_results():
    figure = Figure()
    from bike_fea.geometry import build_geometry
    from tests.test_geometry import vertical_frame

    geometry = build_geometry(vertical_frame())
    draw_frame(figure, geometry, None, warning="Chainstay is shorter than BB drop.")
    assert figure.axes[0].lines
