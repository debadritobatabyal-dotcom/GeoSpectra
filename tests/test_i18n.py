import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import i18n
from i18n import TRANSLATIONS, SUPPORTED_LANGUAGES, t, set_language, get_active_language
import theme

def test_supported_languages():
    assert "en" in SUPPORTED_LANGUAGES
    assert "hi" in SUPPORTED_LANGUAGES
    assert SUPPORTED_LANGUAGES["en"]["name"] == "English"
    assert SUPPORTED_LANGUAGES["hi"]["name"] == "Hindi"

def test_translation_key_parity():
    en_keys = set(TRANSLATIONS["en"].keys())
    hi_keys = set(TRANSLATIONS["hi"].keys())

    missing_in_hi = en_keys - hi_keys
    missing_in_en = hi_keys - en_keys

    assert not missing_in_hi, f"Keys present in EN but missing in HI: {missing_in_hi}"
    assert not missing_in_en, f"Keys present in HI but missing in EN: {missing_in_en}"

def test_non_empty_translations():
    for lang, dictionary in TRANSLATIONS.items():
        for key, val in dictionary.items():
            assert isinstance(val, str), f"Translation for {lang}.{key} must be a string"
            assert len(val.strip()) > 0, f"Translation for {lang}.{key} is empty"

def test_i18n_formatting_and_fallback():

    set_language("en")
    assert get_active_language() == "en"
    formatted = t("percentile_rank", pct=94.2)
    assert "94.2th Percentile" in formatted

    set_language("hi")
    assert get_active_language() == "hi"
    formatted_hi = t("percentile_rank", pct=94.2)
    assert "94.2वाँ" in formatted_hi

    assert t("non_existent_telemetry_key_xyz") == "non_existent_telemetry_key_xyz"

    set_language("en")
    assert get_active_language() == "en"

def test_theme_color_tokens_and_hierarchy():
    for mode in ["light", "dark"]:
        tokens = theme.get_theme_tokens(mode)

        assert "accent_copper" in tokens
        assert "accent_amber" in tokens
        assert "accent_red" in tokens
        assert "accent_blue" in tokens
        assert "accent_green" in tokens

        assert tokens["text_primary"].startswith("#")
        assert tokens["text_secondary"].startswith("#")
        assert tokens["accent_copper"].startswith("#")

def test_frontend_renders_in_hindi():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception on initial load: {at.exception}"

    at.button[1].click().run()
    assert not at.exception, f"App raised exception after login: {at.exception}"

    lang_radios = [r for r in at.radio if r.key == "top_lang_selector"]
    assert len(lang_radios) == 1, "Global language selector radio element should exist"

    lang_radios[0].set_value("हिन्दी").run()
    assert not at.exception, f"App raised exception after switching to Hindi: {at.exception}"

    assert at.session_state["language"] == "hi"

    all_markdown_text = " ".join([m.value for m in at.markdown])
    assert "अन्वेषण" in all_markdown_text or "जियोस्पेक्ट्रा" in all_markdown_text
    assert "प्रॉस्पेक्टिविटी" in all_markdown_text
