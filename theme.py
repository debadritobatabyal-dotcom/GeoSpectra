import streamlit as st

THEMES = {
    "light": {
        "name": "light",
        "background": "#F4F1EA",
        "background_gradient": (
            "radial-gradient(at 0% 0%, rgba(180, 119, 69, 0.08) 0px, transparent 55%), "
            "radial-gradient(at 100% 15%, rgba(36, 92, 115, 0.06) 0px, transparent 50%), "
            "radial-gradient(at 50% 100%, rgba(180, 119, 69, 0.04) 0px, transparent 60%)"
        ),
        "background_secondary": "#EAE6DD",
        "surface": "#FAF9F5",
        "surface_card": "#FAF9F5",
        "surface_elevated": "#FFFFFF",
        "text_primary": "#171917",
        "text_secondary": "#686A65",
        "text_muted": "#8A8B85",
        "border": "#D8D5CC",
        "border_subtle": "#E5E2D9",
        "accent_blue": "#245C73",
        "accent_green": "#567A5B",
        "accent_copper": "#B47745",
        "accent_amber": "#d97706",
        "accent_red": "#c53030",
        "accent_warm": "#B47745",
        "accent_warm_hover": "#9c6232",
        "accent_warm_subtle": "rgba(180, 119, 69, 0.12)",
        "glass_bg": "rgba(250, 249, 245, 0.82)",
        "glass_border": "rgba(36, 92, 115, 0.20)",
        "glass_glow": "0 16px 40px -8px rgba(23, 25, 23, 0.08), 0 0 0 1px rgba(36, 92, 115, 0.12)",
        "btn_glow": "0 4px 14px rgba(36, 92, 115, 0.28)",
        "card_shadow": "0 4px 20px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.03)",
        "card_shadow_elevated": "0 8px 30px rgba(0, 0, 0, 0.08)",
        "btn_primary_bg": "linear-gradient(135deg, #B47745 0%, #8c5324 100%)",
        "btn_primary_border": "#9c6232",
        "btn_primary_color": "#ffffff",
        "btn_primary_shadow": "0 4px 14px rgba(180, 119, 69, 0.35)",
        "btn_secondary_bg": "#FAF9F5",
        "btn_secondary_border": "#D8D5CC",
        "btn_secondary_color": "#171917",
        "btn_secondary_hover_bg": "#EAE6DD",
        "btn_secondary_hover_border": "#B47745",
        "btn_secondary_hover_color": "#B47745",
        "tab_active_color": "#B47745",
        "tab_active_border": "#B47745",
        "tab_inactive_color": "#686A65",
        "input_bg": "#FAF9F5",
        "input_border": "#D8D5CC",
        "input_color": "#171917",
        "expander_bg": "#FAF9F5",
        "expander_border": "#D8D5CC",
        "expander_color": "#171917",
        "plotly_template": "plotly_white",
        "plotly_bg": "rgba(250, 249, 245, 0.7)",
        "plotly_paper_bg": "rgba(250, 249, 245, 0.7)",
        "plotly_grid": "#E5E2D9",
        "plotly_font": "#171917",
        "plotly_line_primary": "#B47745",
        "plotly_line_target": "#686A65",
        "plotly_bar_positive": "#567A5B",
        "plotly_bar_negative": "#c53030",
        "map_frame_border": "#D8D5CC",
        "map_frame_shadow": "0 4px 20px rgba(0, 0, 0, 0.08)",
        "risk_low": "#567A5B",
        "risk_med": "#B47745",
        "risk_high": "#c53030",
        "risk_low_bg": "rgba(86, 122, 91, 0.12)",
        "risk_low_border": "#567A5B",
        "risk_low_text": "#2e5233",
        "risk_med_bg": "rgba(180, 119, 69, 0.12)",
        "risk_med_border": "#B47745",
        "risk_med_text": "#8c5324",
        "risk_high_bg": "rgba(197, 48, 48, 0.10)",
        "risk_high_border": "#c53030",
        "risk_high_text": "#9b1c1c",
        "badge_vhigh_bg": "rgba(180, 119, 69, 0.14)",
        "badge_vhigh_text": "#8c5324",
        "badge_vhigh_border": "#B47745",
        "badge_high_bg": "rgba(217, 119, 6, 0.14)",
        "badge_high_text": "#b45309",
        "badge_high_border": "#d97706",
        "badge_mod_bg": "rgba(202, 138, 4, 0.14)",
        "badge_mod_text": "#a16207",
        "badge_mod_border": "#ca8a04",
        "badge_low_bg": "rgba(86, 122, 91, 0.14)",
        "badge_low_text": "#2e5233",
        "badge_low_border": "#567A5B",
        "badge_vlow_bg": "rgba(36, 92, 115, 0.14)",
        "badge_vlow_text": "#245C73",
        "badge_vlow_border": "#245C73",
        "badge_ood_bg": "rgba(220, 38, 38, 0.12)",
        "badge_ood_text": "#991b1b",
        "badge_ood_border": "#dc2626",
        "driver_item_bg": "#EAE6DD",
        "driver_item_border": "#D8D5CC",
        "driver_item_text": "#171917",
        "track_bg": "#EAE6DD",
        "recovery_box_bg": "#f2f7f3",
        "recovery_box_border": "rgba(86, 122, 91, 0.4)",
        "hazard_box_bg": "#fdf3f3",
        "hazard_box_border": "rgba(197, 48, 48, 0.35)",
    },
    "dark": {
        "name": "dark",
        "background": "#07090e",
        "background_gradient": (
            "radial-gradient(at 0% 0%, rgba(255, 107, 0, 0.12) 0px, transparent 55%), "
            "radial-gradient(at 100% 15%, rgba(230, 81, 0, 0.08) 0px, transparent 50%), "
            "radial-gradient(at 50% 100%, rgba(255, 107, 0, 0.05) 0px, transparent 60%)"
        ),
        "background_secondary": "#0a0e17",
        "surface": "#0d121d",
        "surface_card": "#0d121d",
        "surface_elevated": "#111726",
        "text_primary": "#ffffff",
        "text_secondary": "#e2e8f0",
        "text_muted": "#94a3b8",
        "border": "rgba(255, 255, 255, 0.08)",
        "border_subtle": "rgba(255, 255, 255, 0.05)",
        "accent_blue": "#38bdf8",
        "accent_green": "#22c55e",
        "accent_copper": "#d97736",
        "accent_amber": "#f59e0b",
        "accent_red": "#ef4444",
        "accent_warm": "#ff6b00",
        "accent_warm_hover": "#ff7a1a",
        "accent_warm_subtle": "rgba(255, 107, 0, 0.15)",
        "glass_bg": "rgba(13, 18, 29, 0.74)",
        "glass_border": "rgba(56, 189, 248, 0.22)",
        "glass_glow": "0 16px 44px -8px rgba(0, 0, 0, 0.65), 0 0 24px rgba(56, 189, 248, 0.14)",
        "btn_glow": "0 0 24px rgba(56, 189, 248, 0.35), 0 4px 14px rgba(0, 0, 0, 0.5)",
        "card_shadow": "0 4px 24px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
        "card_shadow_elevated": "0 8px 32px rgba(0, 0, 0, 0.6)",
        "btn_primary_bg": "linear-gradient(135deg, #ff6b00 0%, #e65100 100%)",
        "btn_primary_border": "#ff7a1a",
        "btn_primary_color": "#ffffff",
        "btn_primary_shadow": "0 4px 18px rgba(255, 107, 0, 0.40)",
        "btn_secondary_bg": "#0e131d",
        "btn_secondary_border": "rgba(255, 255, 255, 0.15)",
        "btn_secondary_color": "#f1f5f9",
        "btn_secondary_hover_bg": "#151c2c",
        "btn_secondary_hover_border": "rgba(255, 107, 0, 0.6)",
        "btn_secondary_hover_color": "#ffaa40",
        "tab_active_color": "#ff851a",
        "tab_active_border": "#ff6b00",
        "tab_inactive_color": "#94a3b8",
        "input_bg": "#0e131d",
        "input_border": "rgba(255, 255, 255, 0.14)",
        "input_color": "#ffffff",
        "expander_bg": "#0a0e17",
        "expander_border": "rgba(255, 255, 255, 0.08)",
        "expander_color": "#f1f5f9",
        "plotly_template": "plotly_dark",
        "plotly_bg": "rgba(0,0,0,0)",
        "plotly_paper_bg": "rgba(0,0,0,0)",
        "plotly_grid": "rgba(255, 255, 255, 0.06)",
        "plotly_font": "#cbd5e1",
        "plotly_line_primary": "#ff6b00",
        "plotly_line_target": "#64748b",
        "plotly_bar_positive": "#34d399",
        "plotly_bar_negative": "#ef4444",
        "map_frame_border": "rgba(255, 107, 0, 0.3)",
        "map_frame_shadow": "0 4px 20px rgba(0, 0, 0, 0.6)",
        "risk_low": "#22c55e",
        "risk_med": "#ffaa40",
        "risk_high": "#ef4444",
        "risk_low_bg": "rgba(16, 185, 129, 0.12)",
        "risk_low_border": "#10b981",
        "risk_low_text": "#34d399",
        "risk_med_bg": "rgba(245, 158, 11, 0.12)",
        "risk_med_border": "#f59e0b",
        "risk_med_text": "#fbbf24",
        "risk_high_bg": "rgba(239, 68, 68, 0.12)",
        "risk_high_border": "#ef4444",
        "risk_high_text": "#f87171",
        "badge_vhigh_bg": "rgba(255, 107, 0, 0.18)",
        "badge_vhigh_text": "#ff851a",
        "badge_vhigh_border": "#ff6b00",
        "badge_high_bg": "rgba(245, 158, 11, 0.18)",
        "badge_high_text": "#fbbf24",
        "badge_high_border": "#f59e0b",
        "badge_mod_bg": "rgba(234, 179, 8, 0.15)",
        "badge_mod_text": "#facc15",
        "badge_mod_border": "#eab308",
        "badge_low_bg": "rgba(16, 185, 129, 0.15)",
        "badge_low_text": "#34d399",
        "badge_low_border": "#10b981",
        "badge_vlow_bg": "rgba(56, 189, 248, 0.15)",
        "badge_vlow_text": "#38bdf8",
        "badge_vlow_border": "#0284c7",
        "badge_ood_bg": "rgba(239, 68, 68, 0.15)",
        "badge_ood_text": "#f87171",
        "badge_ood_border": "#ef4444",
        "driver_item_bg": "#111726",
        "driver_item_border": "rgba(255, 255, 255, 0.08)",
        "driver_item_text": "#e2e8f0",
        "track_bg": "rgba(15, 23, 42, 0.6)",
        "recovery_box_bg": "#0d121d",
        "recovery_box_border": "rgba(16, 185, 129, 0.35)",
        "hazard_box_bg": "#0d121d",
        "hazard_box_border": "rgba(239, 68, 68, 0.35)",
    }
}

def get_active_theme():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    return st.session_state["theme"]

def get_theme_tokens(theme_name=None):
    if not theme_name:
        theme_name = get_active_theme()
    return THEMES.get(theme_name, THEMES["light"])

def generate_css(theme_name=None, is_auth=False, b64_bg_map=""):
    t = get_theme_tokens(theme_name)

    if is_auth and b64_bg_map:
        if t.get("is_dark", False) or theme_name == "dark":
            auth_overlay = "radial-gradient(ellipse at center, rgba(10, 15, 29, 0.40) 0%, rgba(10, 15, 29, 0.85) 100%)"
        else:
            auth_overlay = "linear-gradient(135deg, rgba(248, 250, 252, 0.72) 0%, rgba(241, 245, 249, 0.85) 100%)"
        stapp_bg_css = f"""
            background-color: {t['background']} !important;
            background-image: {auth_overlay}, url('data:image/jpeg;base64,{b64_bg_map}') !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        """
    else:
        stapp_bg_css = f"""
            background-color: {t['background']};
            background-image: {t['background_gradient']};
            background-attachment: fixed;
        """

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root, .stApp {{
            --background: {t['background']};
            --background-secondary: {t['background_secondary']};
            --surface: {t['surface']};
            --surface-elevated: {t['surface_elevated']};
            --glass-bg: {t['glass_bg']};
            --glass-border: {t['glass_border']};
            --glass-glow: {t['glass_glow']};
            --btn-glow: {t['btn_glow']};
            --text-primary: {t['text_primary']};
            --text-secondary: {t['text_secondary']};
            --text-muted: {t['text_muted']};
            --border: {t['border']};
            --border-subtle: {t['border_subtle']};
            --accent-blue: {t['accent_blue']};
            --accent-green: {t['accent_green']};
            --accent-copper: {t['accent_copper']};
            --accent-amber: {t['accent_amber']};
            --accent-red: {t['accent_red']};
            --accent-warm: {t['accent_warm']};
            --card-shadow: {t['card_shadow']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-secondary);
        }}

        .stApp {{
            {stapp_bg_css}
            color: var(--text-secondary);
        }}

        /* Remove default Streamlit header bar */
        [data-testid="stHeader"] {{
            background: transparent !important;
            box-shadow: none !important;
            border-bottom: none !important;
        }}
        [data-testid="stToolbar"] {{
            display: none !important;
        }}
        .stApp > header {{
            background: transparent !important;
        }}

        .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1400px !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 800 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.5px !important;
        }}

        label {{
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            color: var(--text-muted) !important;
        }}

        p {{
            color: var(--text-muted);
            line-height: 1.6;
        }}

        /* Buttons: Architectural, Pill, Glowing Cyan/Blue Accents */
        .stButton button {{
            border-radius: 9999px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.6px !important;
            height: 44px !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}

        /* Primary Button */
        .stButton button[kind="primary"],
        button[data-testid="stBaseButton-primary"],
        .stButton button[data-testid="stBaseButton-primary"] {{
            background: {t['btn_primary_bg']} !important;
            border: 1px solid {t['btn_primary_border']} !important;
            color: {t['btn_primary_color']} !important;
            box-shadow: {t['btn_glow']} !important;
        }}
        .stButton button[kind="primary"] *,
        button[data-testid="stBaseButton-primary"] *,
        .stButton button[data-testid="stBaseButton-primary"] * {{
            color: {t['btn_primary_color']} !important;
            fill: {t['btn_primary_color']} !important;
        }}
        .stButton button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {{
            opacity: 0.95 !important;
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.5), 0 6px 20px rgba(0,0,0,0.4) !important;
        }}

        /* Secondary Button */
        .stButton button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"],
        .stButton button[data-testid="stBaseButton-secondary"] {{
            background: {t['glass_bg']} !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid var(--border) !important;
            color: {t['btn_secondary_color']} !important;
        }}
        .stButton button[kind="secondary"] *,
        button[data-testid="stBaseButton-secondary"] *,
        .stButton button[data-testid="stBaseButton-secondary"] * {{
            color: {t['btn_secondary_color']} !important;
        }}
        .stButton button[kind="secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover {{
            background: {t['btn_secondary_hover_bg']} !important;
            border-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
            transform: translateY(-1px);
        }}

        /* Download Buttons */
        .stDownloadButton button {{
            background: {t['glass_bg']} !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid var(--border) !important;
            color: {t['btn_secondary_color']} !important;
            border-radius: 9999px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            transition: all 0.2s ease-in-out !important;
        }}
        .stDownloadButton button:hover {{
            background: {t['btn_secondary_hover_bg']} !important;
            border-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
        }}

        /* Top Navigation: Translucent Minimal Navigation */
        .stTabs [data-baseweb="tab-list"] {{
            display: flex !important;
            width: 100% !important;
            justify-content: space-between !important;
            gap: 12px !important;
            border-bottom: 1px solid var(--border) !important;
            margin-bottom: 24px !important;
            padding-bottom: 0 !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            flex: 1 1 0px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 14px 10px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.12rem !important;
            letter-spacing: 0.4px !important;
            color: {t['tab_inactive_color']} !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
            white-space: nowrap !important;
        }}
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.12rem !important;
            font-weight: 700 !important;
            line-height: 1.25 !important;
            text-align: center !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: var(--text-primary) !important;
            background: {t['accent_warm_subtle']} !important;
            border-radius: 8px 8px 0 0 !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--text-primary) !important;
            font-weight: 800 !important;
            border-bottom: 3px solid var(--accent-blue) !important;
            background: transparent !important;
        }}
        .stTabs [aria-selected="true"] p,
        .stTabs [aria-selected="true"] span {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.12rem !important;
            font-weight: 800 !important;
            color: var(--text-primary) !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: var(--accent-blue) !important;
            height: 3px !important;
        }}
        @media (max-width: 900px) {{
            .stTabs [data-baseweb="tab-list"] {{
                overflow-x: auto !important;
                flex-wrap: nowrap !important;
            }}
            .stTabs [data-baseweb="tab"] {{
                flex: 0 0 auto !important;
                padding: 10px 14px !important;
            }}
        }}

        /* Floating Intelligence Panels & Glass Containers */
        .editorial-card,
        .clean-card,
        .geo-panel {{
            background: var(--glass-bg);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 24px 28px;
            box-shadow: var(--glass-glow);
            margin-bottom: 20px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .geo-panel:hover {{
            border-color: var(--accent-blue);
        }}

        /* Authentication Glass Card */
        .geo-auth-glass,
        .geo-auth-card,
        div[class*="st-key-auth_card"] {{
            max-width: 470px !important;
            margin: 12px auto 16px auto !important;
            background: var(--glass-bg) !important;
            backdrop-filter: blur(28px) saturate(190%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 22px !important;
            padding: 32px 34px 26px 34px !important;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35), var(--glass-glow) !important;
            text-align: center !important;
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="st-key-auth_card"]) {{
            background: var(--glass-bg) !important;
            backdrop-filter: blur(28px) saturate(190%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 22px !important;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35), var(--glass-glow) !important;
            max-width: 470px !important;
            margin: 12px auto 16px auto !important;
        }}
        .geo-auth-card:hover,
        div[class*="st-key-auth_card"]:hover {{
            box-shadow: 0 0 35px rgba(56, 189, 248, 0.25), 0 28px 65px rgba(0, 0, 0, 0.4) !important;
        }}
        div[class*="st-key-auth_card"] [data-testid="stTextInput"] label {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
            text-align: left !important;
            display: block !important;
            margin-bottom: 4px !important;
        }}
        div[class*="st-key-auth_card"] [data-testid="stTextInput"] input {{
            border-radius: 12px !important;
            font-size: 0.92rem !important;
            padding: 10px 14px !important;
        }}

        /* Authentication Hero Container & Ambient Exploration Backdrop */
        .geo-auth-hero-backdrop {{
            position: relative;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 10px 0 20px 0;
            overflow: hidden;
        }}

        .geo-auth-ambient-map {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-size: cover;
            background-position: center;
            opacity: 0.18;
            filter: contrast(125%) saturate(140%);
            pointer-events: none;
            z-index: 0;
            border-radius: 16px;
            animation: subtleMapPulse 24s ease-in-out infinite alternate;
        }}

        @keyframes subtleMapPulse {{
            0% {{ transform: scale(1.0); opacity: 0.15; }}
            50% {{ transform: scale(1.02); opacity: 0.20; }}
            100% {{ transform: scale(1.0); opacity: 0.15; }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            .geo-auth-ambient-map {{
                animation: none !important;
            }}
        }}

        /* OTP 6-Digit Display Cells */
        .geo-otp-container {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 14px 0 18px 0;
        }}
        .geo-otp-box {{
            width: 46px;
            height: 52px;
            border-radius: 10px;
            background: var(--surface);
            border: 1.5px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Space Grotesk', monospace;
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-primary);
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
            transition: all 0.2s ease;
        }}
        .geo-otp-box.filled {{
            border-color: var(--accent-copper);
            box-shadow: 0 0 10px rgba(180, 119, 69, 0.3);
            color: var(--accent-copper);
        }}
        .geo-otp-box.active {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
        }}

        /* Subtle Contextual Metadata Strip (Section 12) */
        .geo-auth-meta-strip {{
            position: relative;
            z-index: 2;
            width: 100%;
            max-width: 820px;
            margin: 16px auto 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
        }}
        .geo-meta-pill {{
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 10px 14px;
            text-align: left;
            transition: border-color 0.2s ease;
        }}
        .geo-meta-pill:hover {{
            border-color: var(--accent-copper);
        }}

        /* Score Display: Dominant Editorial Number */
        .score-number,
        .geo-kpi-hero {{
            font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
            font-size: 4.2rem;
            font-weight: 800;
            letter-spacing: -2.5px;
            line-height: 0.95;
            color: var(--text-primary) !important;
        }}

        .badge-status,
        .geo-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }}
        .badge-vhigh {{
            background: {t['badge_vhigh_bg']};
            color: {t['badge_vhigh_text']} !important;
            border: 1px solid {t['badge_vhigh_border']};
        }}
        .badge-high {{
            background: {t['badge_high_bg']};
            color: {t['badge_high_text']} !important;
            border: 1px solid {t['badge_high_border']};
        }}
        .badge-mod {{
            background: {t['badge_mod_bg']};
            color: {t['badge_mod_text']} !important;
            border: 1px solid {t['badge_mod_border']};
        }}
        .badge-low {{
            background: {t['badge_low_bg']};
            color: {t['badge_low_text']} !important;
            border: 1px solid {t['badge_low_border']};
        }}
        .badge-vlow {{
            background: {t['badge_vlow_bg']};
            color: {t['badge_vlow_text']} !important;
            border: 1px solid {t['badge_vlow_border']};
        }}
        .badge-ood {{
            background: {t['badge_ood_bg']};
            color: {t['badge_ood_text']} !important;
            border: 1px solid {t['badge_ood_border']};
        }}

        /* Driver Item */
        .driver-item {{
            font-size: 0.80rem;
            font-weight: 500;
            color: {t['driver_item_text']} !important;
            padding: 6px 12px;
            background: {t['driver_item_bg']};
            border-radius: 4px;
            border: 1px solid {t['driver_item_border']};
            margin-bottom: 6px;
        }}

        /* Form widgets overrides - Comprehensive for Text, Number, and Select widgets */
        div[data-baseweb="input"],
        div[data-baseweb="base-input"],
        div[data-testid="stNumberInputContainer"],
        .stNumberInput div[data-baseweb="input"],
        .stTextInput div[data-baseweb="input"] {{
            background-color: {t['input_bg']} !important;
            border: 1px solid {t['input_border']} !important;
            border-radius: 6px !important;
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        div[data-testid="stNumberInputContainer"] input,
        input[data-testid="stNumberInputField"],
        .stTextInput input,
        .stNumberInput input {{
            color: {t['input_color']} !important;
            -webkit-text-fill-color: {t['input_color']} !important;
            background-color: transparent !important;
            font-family: 'Space Grotesk', monospace, sans-serif !important;
            font-weight: 600 !important;
        }}
        /* Stepper buttons (+/-) on number inputs */
        div[data-testid="stNumberInputContainer"] button,
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {{
            background-color: transparent !important;
            color: {t['text_secondary']} !important;
            border: none !important;
        }}
        div[data-testid="stNumberInputContainer"] button svg,
        button[data-testid="stNumberInputStepDown"] svg,
        button[data-testid="stNumberInputStepUp"] svg {{
            fill: {t['text_secondary']} !important;
            stroke: {t['text_secondary']} !important;
        }}
        div[data-testid="stNumberInputContainer"] button:hover,
        button[data-testid="stNumberInputStepDown"]:hover,
        button[data-testid="stNumberInputStepUp"]:hover {{
            background-color: {t['accent_warm_subtle']} !important;
            color: {t['accent_warm']} !important;
        }}
        div[data-testid="stNumberInputContainer"] button:hover svg,
        button[data-testid="stNumberInputStepDown"]:hover svg,
        button[data-testid="stNumberInputStepUp"]:hover svg {{
            fill: {t['accent_warm']} !important;
            stroke: {t['accent_warm']} !important;
        }}
        /* Widget labels */
        label,
        label p,
        [data-testid="stWidgetLabel"] p {{
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            color: var(--text-muted) !important;
        }}
        /* Placeholders */
        input::placeholder,
        textarea::placeholder {{
            color: {t['text_muted']} !important;
            -webkit-text-fill-color: {t['text_muted']} !important;
            opacity: 1 !important;
        }}
        /* Study Domain Dropdown Container & Control */
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-testid="stSelectbox"],
        .element-container:has(.geospectra-domain-anchor) + .element-container div[data-testid="stSelectbox"] {{
            width: 100% !important;
            max-width: 350px !important;
        }}
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] > div,
        .element-container:has(.geospectra-domain-anchor) + .element-container div[data-baseweb="select"] > div {{
            background-color: {t['surface']} !important;
            border: 1px solid {t['border']} !important;
            border-radius: 9999px !important;
            min-height: 38px !important;
            height: 38px !important;
            padding: 0 14px 0 16px !important;
            display: flex !important;
            align-items: center !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            cursor: pointer !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] > div:hover,
        .element-container:has(.geospectra-domain-anchor) + .element-container div[data-baseweb="select"] > div:hover {{
            border-color: rgba(255, 107, 0, 0.45) !important;
            box-shadow: 0 2px 10px rgba(255, 107, 0, 0.12) !important;
        }}
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] > div:focus-within,
        .element-container:has(.geospectra-domain-anchor) + .element-container div[data-baseweb="select"] > div:focus-within {{
            border-color: {t['accent_warm']} !important;
            box-shadow: 0 0 0 1px {t['accent_warm']}, 0 2px 10px rgba(255, 107, 0, 0.20) !important;
        }}
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] div[role="combobox"],
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] span,
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] div,
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] p,
        .element-container:has(.geospectra-domain-anchor) + .element-container div[data-baseweb="select"] span {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.8px !important;
            text-transform: uppercase !important;
            color: {t['text_primary']} !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] svg,
        .element-container:has(.geospectra-domain-anchor) + .element-container div[data-baseweb="select"] svg {{
            fill: {t['text_secondary']} !important;
            width: 15px !important;
            height: 15px !important;
            opacity: 0.8 !important;
            transition: fill 0.2s ease, opacity 0.2s ease !important;
        }}
        div[data-testid="stColumn"]:has(.geospectra-domain-anchor) div[data-baseweb="select"] > div:hover svg,
        .element-container:has(.geospectra-domain-anchor) + .element-container div[data-baseweb="select"] > div:hover svg {{
            fill: {t['accent_warm']} !important;
            opacity: 1 !important;
        }}

        /* General Selectbox styling */
        div[data-baseweb="select"] > div {{
            background-color: {t['input_bg']} !important;
            border: 1px solid {t['input_border']} !important;
            color: {t['input_color']} !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] p {{
            color: {t['input_color']} !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: {t['text_secondary']} !important;
        }}

        /* Floating Popover Dropdown Menu */
        div[data-baseweb="popover"] {{
            background-color: {t['surface']} !important;
            border: 1px solid {t['border']} !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45), 0 2px 8px rgba(0, 0, 0, 0.15) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin-top: 4px !important;
        }}
        div[data-baseweb="menu"],
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] ul[role="listbox"],
        div[data-baseweb="popover"] ul[data-testid="stSelectboxVirtualDropdown"] {{
            background: transparent !important;
            background-color: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
        }}
        div[data-baseweb="popover"] li[role="option"] {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.8px !important;
            text-transform: uppercase !important;
            color: {t['text_primary']} !important;
            box-sizing: border-box !important;
            height: 40px !important;
            line-height: 40px !important;
            padding: 0 16px !important;
            margin: 0 !important;
            cursor: pointer !important;
            transition: background-color 0.15s ease, color 0.15s ease !important;
            display: flex !important;
            align-items: center !important;
        }}
        div[data-baseweb="popover"] li[role="option"]:hover,
        div[data-baseweb="popover"] li[role="option"][aria-highlighted="true"] {{
            background-color: {t['accent_warm_subtle']} !important;
            color: {t['accent_warm']} !important;
        }}
        div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {{
            background-color: {t['accent_warm_subtle']} !important;
            color: {t['accent_warm']} !important;
            border-left: 3px solid {t['accent_warm']} !important;
            font-weight: 800 !important;
            padding-left: 13px !important;
            padding-right: 32px !important;
        }}
        div[data-baseweb="popover"] li[role="option"][aria-selected="true"]::after {{
            content: "✓";
            position: absolute;
            right: 16px;
            color: {t['accent_warm']};
            font-weight: 900;
            font-size: 0.82rem;
        }}
        [data-testid="stExpander"] {{
            background-color: {t['expander_bg']} !important;
            border: 1px solid {t['expander_border']} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stExpander"] summary {{
            color: {t['expander_color']} !important;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 9999px !important;
            padding: 2px 6px !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 2px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
        }}
        div[data-testid="stRadio"] label {{
            margin-bottom: 0 !important;
            padding: 2px 8px !important;
            border-radius: 9999px !important;
            cursor: pointer !important;
            transition: background 0.2s ease !important;
        }}
        div[data-testid="stRadio"] label:hover {{
            background: rgba(56, 189, 248, 0.12) !important;
        }}
        div[data-testid="stRadio"] label span {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            color: var(--text-secondary) !important;
        }}

        /* Segmented Control / Language Switcher */
        div[data-testid="stSegmentedControl"] {{
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            padding: 2px !important;
        }}
        div[data-testid="stSegmentedControl"] button {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            border-radius: 4px !important;
            border: none !important;
            color: var(--text-secondary) !important;
            background: transparent !important;
            padding: 4px 10px !important;
            min-height: 28px !important;
        }}
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
            background-color: var(--accent-blue) !important;
            color: #ffffff !important;
        }}

        /* Folium Map Frame */
        iframe {{
            border-radius: 8px !important;
            border: 1px solid {t['map_frame_border']} !important;
            box-shadow: {t['map_frame_shadow']} !important;
        }}
    </style>
    """
    return css
