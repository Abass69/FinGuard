from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_dashboard_research_view():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'dashboard.py')).run(timeout=30)
    assert not app.exception
    app.sidebar.radio[0].set_value('Research evaluation').run(timeout=30)
    assert not app.exception
    assert len(app.dataframe)==1
