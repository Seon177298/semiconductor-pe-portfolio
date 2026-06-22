from __future__ import annotations

from pathlib import Path


def test_streamlit_ui_uses_api_client_instead_of_direct_csv_reads() -> None:
    ui_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    source = ui_path.read_text(encoding="utf-8")

    assert "requests." in source
    assert "pd.read_csv" not in source
    assert "read_csv(" not in source


def test_streamlit_threshold_slider_previews_metrics_before_policy_apply() -> None:
    ui_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    source = ui_path.read_text(encoding="utf-8")

    slider_index = source.index('st.sidebar.slider("Defect threshold"')
    metrics_index = source.index('api_get("/quality/metrics"', slider_index)
    policy_index = source.index('api_post("/quality/threshold"', slider_index)

    assert slider_index < metrics_index < policy_index


def test_streamlit_failure_cases_tab_uses_failure_cases_endpoint() -> None:
    ui_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    source = ui_path.read_text(encoding="utf-8")

    failure_tab_index = source.index("with failure_tab:")
    failure_endpoint_index = source.index('api_get("/quality/failure-cases"', failure_tab_index)
    prediction_endpoint_index = source.find('api_get("/quality/predictions"', failure_tab_index)

    assert failure_endpoint_index > failure_tab_index
    assert prediction_endpoint_index == -1 or prediction_endpoint_index > source.index("with vision_tab:")


def test_streamlit_surfaces_policy_and_synthetic_operations_context() -> None:
    ui_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    source = ui_path.read_text(encoding="utf-8")

    assert 'api_get("/quality/policies"' in source
    assert 'api_get("/operations/events"' in source
    assert "Synthetic operations context" in source


def test_streamlit_has_digital_thread_tab_and_uses_only_api_endpoints() -> None:
    ui_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    source = ui_path.read_text(encoding="utf-8")

    digital_tab_index = source.index("with digital_thread_tab:")
    assert '"Digital Thread"' in source
    assert "Public case-informed synthetic layer, not internal company internal data." in source
    assert 'api_get("/digital-thread/source-map"' in source
    assert 'api_get("/digital-thread/lots"' in source
    assert 'api_get(f"/digital-thread/lot/{selected_lot_id}"' in source
    assert 'api_get("/digital-thread/trace"' in source
    assert source.index('api_get("/digital-thread/trace"', digital_tab_index) > digital_tab_index


def test_streamlit_failure_cases_mentions_digital_thread_trace() -> None:
    ui_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    source = ui_path.read_text(encoding="utf-8")

    failure_tab_index = source.index("with failure_tab:")
    digital_hint_index = source.index("Digital Thread tab", failure_tab_index)

    assert digital_hint_index > failure_tab_index
