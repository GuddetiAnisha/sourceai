from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'

def test_app_load_and_scenario_changes():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 6
    app.slider(key='weight_cost').set_value(80)
    app.slider(key='maturity_Data').set_value(5)
    app.selectbox[0].set_value('Managed services')
    app.run()
    assert not app.exception
    assert app.metric[0].value == '6'
    app.selectbox[0].set_value('Field services').run()
    assert not app.exception

def test_zero_weights_shows_helpful_warning():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    for key in ['cost', 'quality', 'delivery', 'risk', 'sustainability', 'ai_readiness']:
        app.slider(key=f'weight_{key}').set_value(0)
    app.run()
    assert not app.exception
    assert any('total more than zero' in item.value for item in app.warning)
