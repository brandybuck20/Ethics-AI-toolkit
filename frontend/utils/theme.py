import streamlit as st
from pathlib import Path
import json
from typing import Dict, Any, Optional, List

class ThemeManager:
    """Theme management utilities for AI Ethics Toolkit"""
    
    # Available themes
    THEMES = {
        'light': {
            'name': 'Light',
            'primary_color': '#2E5AAC',
            'background_color': '#FFFFFF',
            'secondary_background_color': '#F7F9FC',
            'text_color': '#1E293B'
        },
        'dark': {
            'name': 'Dark',
            'primary_color': '#60A5FA',
            'background_color': '#1F2937',
            'secondary_background_color': '#111827',
            'text_color': '#F9FAFB'
        },
        'high_contrast': {
            'name': 'High Contrast',
            'primary_color': '#0000FF',
            'background_color': '#FFFFFF',
            'secondary_background_color': '#F0F0F0',
            'text_color': '#000000'
        },
        'blue': {
            'name': 'Blue',
            'primary_color': '#1E40AF',
            'background_color': '#FFFFFF',
            'secondary_background_color': '#EFF6FF',
            'text_color': '#1E293B'
        },
        'green': {
            'name': 'Green',
            'primary_color': '#059669',
            'background_color': '#FFFFFF',
            'secondary_background_color': '#ECFDF5',
            'text_color': '#1E293B'
        }
    }
    
    @staticmethod
    def load_custom_css():
        """Load all custom stylesheets"""
        
        # Define CSS file paths
        css_files = [
            'frontend/styles/main.css',
            'frontend/styles/components.css',
            'frontend/styles/themes.css'
        ]
        
        # Load each CSS file
        for css_file in css_files:
            ThemeManager._load_css_file(css_file)
        
        # Apply current theme
        current_theme = st.session_state.get('theme', 'light')
        ThemeManager.apply_theme(current_theme)

    @staticmethod
    def _load_css_file(css_file_path: str):
        """Load individual CSS file"""
        
        css_path = Path(css_file_path)
        
        if css_path.exists():
            try:
                with open(css_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading CSS file {css_file_path}: {str(e)}")
        else:
            st.warning(f"CSS file not found: {css_file_path}")

    @staticmethod
    def apply_theme(theme_name: str):
        """Apply specific theme"""
        
        if theme_name not in ThemeManager.THEMES:
            theme_name = 'light'
        
        theme_config = ThemeManager.THEMES[theme_name]
        
        # Generate theme CSS
        theme_css = f"""
        <style>
        :root {{
            --theme-primary: {theme_config['primary_color']};
            --theme-background: {theme_config['background_color']};
            --theme-secondary-bg: {theme_config['secondary_background_color']};
            --theme-text: {theme_config['text_color']};
        }}
        
        [data-theme="{theme_name}"] {{
            --primary-color: {theme_config['primary_color']};
            --bg-primary: {theme_config['background_color']};
            --bg-secondary: {theme_config['secondary_background_color']};
            --text-primary: {theme_config['text_color']};
        }}
        
        .stApp {{
            background-color: {theme_config['background_color']};
            color: {theme_config['text_color']};
        }}
        
        .stSidebar > div {{
            background-color: {theme_config['secondary_background_color']};
        }}
        </style>
        """
        
        st.markdown(theme_css, unsafe_allow_html=True)
        
        # Update session state
        st.session_state.theme = theme_name

    @staticmethod
    def get_available_themes() -> Dict[str, str]:
        """Get list of available themes"""
        
        return {key: config['name'] for key, config in ThemeManager.THEMES.items()}

    @staticmethod
    def get_current_theme() -> str:
        """Get current theme name"""
        
        return st.session_state.get('theme', 'light')

    @staticmethod
    def render_theme_selector():
        """Render theme selection widget"""
        
        st.markdown("#### 🎨 Theme Selection")
        
        current_theme = ThemeManager.get_current_theme()
        available_themes = ThemeManager.get_available_themes()
        
        # Get current theme index
        theme_keys = list(available_themes.keys())
        try:
            current_index = theme_keys.index(current_theme)
        except ValueError:
            current_index = 0
        
        # Theme selector
        selected_theme = st.selectbox(
            "Choose Theme",
            options=theme_keys,
            format_func=lambda x: available_themes[x],
            index=current_index,
            key="theme_selector"
        )
        
        # Apply theme if changed
        if selected_theme != current_theme:
            ThemeManager.apply_theme(selected_theme)
            st.rerun()

    @staticmethod
    def create_custom_theme(name: str, config: Dict[str, str]) -> bool:
        """Create custom theme"""
        
        required_keys = ['primary_color', 'background_color', 'secondary_background_color', 'text_color']
        
        # Validate config
        if not all(key in config for key in required_keys):
            return False
        
        # Add name
        config['name'] = name
        
        # Add to themes
        ThemeManager.THEMES[name.lower().replace(' ', '_')] = config
        
        return True

    @staticmethod
    def export_theme_config() -> str:
        """Export current theme configuration"""
        
        current_theme = ThemeManager.get_current_theme()
        theme_config = ThemeManager.THEMES.get(current_theme, {})
        
        export_data = {
            'theme_name': current_theme,
            'config': theme_config,
            'exported_at': str(pd.Timestamp.now())
        }
        
        return json.dumps(export_data, indent=2)

    @staticmethod
    def import_theme_config(json_data: str) -> bool:
        """Import theme configuration"""
        
        try:
            theme_data = json.loads(json_data)
            
            if 'config' not in theme_data:
                return False
            
            config = theme_data['config']
            theme_name = theme_data.get('theme_name', 'imported_theme')
            
            return ThemeManager.create_custom_theme(theme_name, config)
            
        except json.JSONDecodeError:
            return False

    @staticmethod
    def get_theme_preview_css(theme_name: str) -> str:
        """Generate preview CSS for theme"""
        
        if theme_name not in ThemeManager.THEMES:
            return ""
        
        theme_config = ThemeManager.THEMES[theme_name]
        
        preview_css = f"""
        .theme-preview-{theme_name} {{
            background-color: {theme_config['background_color']};
            color: {theme_config['text_color']};
            border: 2px solid {theme_config['primary_color']};
            border-radius: 8px;
            padding: 16px;
            margin: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .theme-preview-{theme_name}:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .theme-preview-{theme_name} .preview-primary {{
            background-color: {theme_config['primary_color']};
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
            margin-top: 8px;
        }}
        """
        
        return preview_css

    @staticmethod
    def render_theme_previews():
        """Render theme preview cards"""
        
        st.markdown("#### 🎨 Theme Previews")
        
        # Generate preview CSS for all themes
        preview_css = ""
        for theme_name in ThemeManager.THEMES.keys():
            preview_css += ThemeManager.get_theme_preview_css(theme_name)
        
        st.markdown(f"<style>{preview_css}</style>", unsafe_allow_html=True)
        
        # Create preview cards
        cols = st.columns(3)
        
        for i, (theme_key, theme_config) in enumerate(ThemeManager.THEMES.items()):
            with cols[i % 3]:
                preview_html = f"""
                <div class="theme-preview-{theme_key}" onclick="selectTheme('{theme_key}')">
                    <h4>{theme_config['name']}</h4>
                    <p>Sample text in this theme</p>
                    <div class="preview-primary">Primary Button</div>
                </div>
                """
                
                st.markdown(preview_html, unsafe_allow_html=True)
                
                if st.button(f"Apply {theme_config['name']}", key=f"apply_{theme_key}"):
                    ThemeManager.apply_theme(theme_key)
                    st.rerun()

# Global theme functions for easy access
def load_custom_css():
    """Global function to load custom CSS"""
    ThemeManager.load_custom_css()

def apply_theme(theme_name: str):
    """Global function to apply theme"""
    ThemeManager.apply_theme(theme_name)

def get_current_theme() -> str:
    """Global function to get current theme"""
    return ThemeManager.get_current_theme()
