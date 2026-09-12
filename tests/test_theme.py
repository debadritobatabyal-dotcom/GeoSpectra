import pytest
import theme
from streamlit.testing.v1 import AppTest
import os

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

def test_theme_keys_parity():
    light_keys = set(theme.THEMES["light"].keys())
    dark_keys = set(theme.THEMES["dark"].keys())
    assert light_keys == dark_keys, f"Mismatched keys: {light_keys.symmetric_difference(dark_keys)}"

def test_light_mode_color_palette():
    t = theme.THEMES["light"]
    assert t["background"] == "#F4F1EA"
    assert t["background_secondary"] == "#EAE6DD"
    assert t["surface"] == "#FAF9F5"
    assert t["text_primary"] == "#171917"
    assert t["text_secondary"] == "#686A65"
    assert t["text_muted"] == "#8A8B85"
    assert t["border"] == "#D8D5CC"
    assert t["accent_blue"] == "#245C73"
    assert t["accent_green"] == "#567A5B"
    assert t["accent_warm"] == "#B47745"

def test_dark_mode_preserved():
    t = theme.THEMES["dark"]
    assert t["background"] == "#07090e"
    assert t["surface"] == "#0d121d"
    assert t["text_primary"] == "#ffffff"
    assert t["accent_warm"] == "#ff6b00"

def test_generate_css_output():
    css_light = theme.generate_css("light")
    assert "--background: #F4F1EA;" in css_light
    assert "--surface: #FAF9F5;" in css_light
    assert ".clean-card" in css_light

    css_dark = theme.generate_css("dark")
    assert "--background: #07090e;" in css_dark
    assert "--surface: #0d121d;" in css_dark
    assert ".clean-card" in css_dark

def test_theme_toggle_apptest():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert at.session_state["theme"] == "light"

    at.button[1].click().run()
    assert at.session_state["authenticated"] is True

    at.session_state["theme"] = "dark"
    at.run()
    assert at.session_state["theme"] == "dark"

    at.session_state["theme"] = "light"
    at.run()
    assert at.session_state["theme"] == "light"

def test_form_inputs_styling_and_contrast():
    css_dark = theme.generate_css("dark")
    assert 'data-testid="stNumberInputContainer"' in css_dark
    assert 'background-color: #0e131d !important;' in css_dark
    assert 'color: #ffffff !important;' in css_dark

    css_light = theme.generate_css("light")
    assert 'data-testid="stNumberInputContainer"' in css_light
    assert 'background-color: #FAF9F5 !important;' in css_light
    assert 'color: #171917 !important;' in css_light

def test_selected_target_contrast_across_themes():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button[1].click().run()

    at.session_state["theme"] = "light"
    at.run()
    markdown_texts_light = [m.value for m in at.markdown]
    assert any("Selected Target" in text and "#171917" in text for text in markdown_texts_light)
    assert not any("Selected Target" in text and "#ffffff" in text for text in markdown_texts_light)

    at.session_state["theme"] = "dark"
    at.run()
    markdown_texts_dark = [m.value for m in at.markdown]
    assert any("Selected Target" in text and "#ffffff" in text for text in markdown_texts_dark)
