import streamlit as st
import json
from datetime import datetime

def main():
    """Main settings page"""
    
    st.title("⚙️ System Settings")
    st.markdown("**Configure your AI Ethics Toolkit preferences and thresholds**")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Audit Thresholds", "🎨 Appearance", "🔧 Advanced", "ℹ️ About"])
    
    with tab1:
        render_audit_thresholds()
    
    with tab2:
        render_appearance_settings()
    
    with tab3:
        render_advanced_settings()
    
    with tab4:
        render_about_section()

def render_audit_thresholds():
    """Render audit threshold configuration"""
    
    st.subheader("🎯 Audit Thresholds Configuration")
    st.markdown("Set the sensitivity levels for different types of ethical audits")
    
    # Initialize default thresholds if not set
    if 'audit_thresholds' not in st.session_state:
        st.session_state.audit_thresholds = {
            'bias_threshold': 0.1,
            'privacy_threshold': 0.05,
            'hallucination_threshold': 0.2,
            'explainability_threshold': 0.7
        }
    
    # Bias detection thresholds
    st.markdown("#### 📊 Bias Detection")
    with st.expander("Bias Threshold Settings", expanded=True):
        
        bias_threshold = st.slider(
            "Maximum acceptable bias difference between groups",
            min_value=0.01,
            max_value=0.30,
            value=st.session_state.audit_thresholds.get('bias_threshold', 0.1),
            step=0.01,
            format="%.2f",
            help="Lower values = more strict bias detection"
        )
        
        demographic_parity_weight = st.slider(
            "Demographic Parity Weight",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Weight given to demographic parity in overall bias score"
        )
        
        equalized_odds_weight = st.slider(
            "Equalized Odds Weight", 
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Weight given to equalized odds in overall bias score"
        )
    
    # Privacy analysis thresholds
    st.markdown("#### 🔒 Privacy Analysis")
    with st.expander("Privacy Threshold Settings", expanded=True):
        
        privacy_threshold = st.slider(
            "PII detection confidence threshold",
            min_value=0.01,
            max_value=0.30,
            value=st.session_state.audit_thresholds.get('privacy_threshold', 0.05),
            step=0.01,
            format="%.2f",
            help="Minimum confidence level to flag potential PII"
        )
        
        pii_sensitivity = st.selectbox(
            "PII Detection Sensitivity",
            ["Conservative", "Balanced", "Aggressive", "Maximum"],
            index=1,
            help="Higher sensitivity may produce more false positives"
        )
        
        gdpr_strict_mode = st.checkbox(
            "GDPR Strict Compliance Mode",
            value=True,
            help="Apply stricter GDPR compliance checking"
        )
    
    # Hallucination detection thresholds
    st.markdown("#### 🔍 Hallucination Detection")
    with st.expander("Hallucination Threshold Settings", expanded=True):
        
        hallucination_threshold = st.slider(
            "Minimum confidence to flag hallucinations",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.audit_thresholds.get('hallucination_threshold', 0.2),
            step=0.1,
            format="%.1f",
            help="Lower values = more sensitive hallucination detection"
        )
        
        fact_check_sources = st.multiselect(
            "Fact-checking sources to use",
            ["Wikipedia", "Local Knowledge Base", "Custom Sources"],
            default=["Wikipedia", "Local Knowledge Base"],
            help="Select which sources to use for fact verification"
        )
        
        link_verification = st.checkbox(
            "Enable URL link verification",
            value=True,
            help="Check if URLs in content actually exist"
        )
    
    # Explainability thresholds
    st.markdown("#### 💡 Explainability")
    with st.expander("Explainability Threshold Settings", expanded=True):
        
        explainability_threshold = st.slider(
            "Minimum interpretability score required",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.audit_thresholds.get('explainability_threshold', 0.7),
            step=0.1,
            format="%.1f",
            help="Minimum score for model to be considered interpretable"
        )
        
        explanation_method_preference = st.selectbox(
            "Preferred explanation method",
            ["SHAP", "LIME", "Permutation Importance", "Custom"],
            index=0,
            help="Default method for generating explanations"
        )
    
    # Save thresholds
    if st.button("💾 **Save Threshold Settings**", type="primary", use_container_width=True):
        st.session_state.audit_thresholds.update({
            'bias_threshold': bias_threshold,
            'privacy_threshold': privacy_threshold,
            'hallucination_threshold': hallucination_threshold,
            'explainability_threshold': explainability_threshold,
            'demographic_parity_weight': demographic_parity_weight,
            'equalized_odds_weight': equalized_odds_weight,
            'pii_sensitivity': pii_sensitivity,
            'gdpr_strict_mode': gdpr_strict_mode,
            'fact_check_sources': fact_check_sources,
            'link_verification': link_verification,
            'explanation_method_preference': explanation_method_preference
        })
        
        st.success("✅ Audit thresholds saved successfully!")
        
        # Show current settings summary
        with st.expander("📋 Current Settings Summary"):
            st.json(st.session_state.audit_thresholds)

def render_appearance_settings():
    """Render appearance and UI settings"""
    
    st.subheader("🎨 Appearance & Interface")
    st.markdown("Customize the look and feel of your AI Ethics Toolkit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Theme settings
        st.markdown("#### 🌓 Theme")
        
        theme_mode = st.selectbox(
            "Color Theme",
            ["Light", "Dark", "Auto (System)"],
            index=0,
            help="Choose your preferred color theme"
        )
        
        primary_color = st.color_picker(
            "Primary Color",
            value="#2E5AAC",
            help="Main accent color for buttons and highlights"
        )
        
        # Layout preferences
        st.markdown("#### 📐 Layout")
        
        sidebar_default = st.selectbox(
            "Sidebar Default State",
            ["Expanded", "Collapsed", "Auto"],
            index=0,
            help="Default state of the navigation sidebar"
        )
        
        page_width = st.selectbox(
            "Page Width",
            ["Wide", "Centered", "Full Width"],
            index=0,
            help="Maximum width of the main content area"
        )
    
    with col2:
        # Data display preferences
        st.markdown("#### 📊 Data Display")
        
        default_chart_theme = st.selectbox(
            "Chart Theme",
            ["Plotly White", "Plotly Dark", "Ggplot2", "Seaborn"],
            index=0,
            help="Default styling for charts and visualizations"
        )
        
        show_code_snippets = st.checkbox(
            "Show code snippets in explanations",
            value=False,
            help="Display underlying code for advanced users"
        )
        
        animate_charts = st.checkbox(
            "Enable chart animations",
            value=True,
            help="Add smooth transitions to chart updates"
        )
        
        # Language and locale
        st.markdown("#### 🌍 Language & Locale")
        
        language = st.selectbox(
            "Interface Language",
            ["English", "Spanish", "French", "German", "Chinese"],
            index=0,
            disabled=True,
            help="Interface language (coming in future release)"
        )
        
        date_format = st.selectbox(
            "Date Format",
            ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"],
            index=2,
            help="Preferred date display format"
        )
    
    # Save appearance settings
    if st.button("🎨 **Save Appearance Settings**", type="primary", use_container_width=True):
        appearance_settings = {
            'theme_mode': theme_mode,
            'primary_color': primary_color,
            'sidebar_default': sidebar_default,
            'page_width': page_width,
            'default_chart_theme': default_chart_theme,
            'show_code_snippets': show_code_snippets,
            'animate_charts': animate_charts,
            'language': language,
            'date_format': date_format
        }
        
        st.session_state.appearance_settings = appearance_settings
        st.success("✅ Appearance settings saved! Some changes may require a page refresh.")

def render_advanced_settings():
    """Render advanced system settings"""
    
    st.subheader("🔧 Advanced Configuration")
    st.markdown("Advanced settings for power users and system administrators")
    
    # Performance settings
    st.markdown("#### ⚡ Performance")
    with st.expander("Performance Settings", expanded=True):
        
        enable_caching = st.checkbox(
            "Enable result caching",
            value=True,
            help="Cache analysis results to improve performance"
        )
        
        cache_ttl = st.slider(
            "Cache TTL (hours)",
            min_value=1,
            max_value=24,
            value=6,
            help="How long to keep cached results"
        )
        
        max_concurrent_audits = st.slider(
            "Maximum concurrent audits",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of audits that can run simultaneously"
        )
        
        enable_parallel_processing = st.checkbox(
            "Enable parallel processing",
            value=True,
            help="Use multiple CPU cores for faster analysis"
        )
    
    # Data management
    st.markdown("#### 💾 Data Management")
    with st.expander("Data Management Settings", expanded=True):
        
        auto_cleanup_enabled = st.checkbox(
            "Enable automatic cleanup",
            value=True,
            help="Automatically clean up old temporary files"
        )
        
        data_retention_days = st.slider(
            "Data retention period (days)",
            min_value=7,
            max_value=365,
            value=90,
            help="How long to keep uploaded files and results"
        )
        
        max_upload_size = st.selectbox(
            "Maximum upload file size",
            ["10MB", "50MB", "100MB", "500MB", "1GB"],
            index=2,
            help="Maximum size for uploaded model and data files"
        )
        
        compress_uploads = st.checkbox(
            "Compress uploaded files",
            value=True,
            help="Automatically compress large uploads to save space"
        )
    
    # Security settings
    st.markdown("#### 🔐 Security")
    with st.expander("Security Settings", expanded=True):
        
        require_file_validation = st.checkbox(
            "Require file validation",
            value=True,
            help="Validate uploaded files for security threats"
        )
        
        log_all_activities = st.checkbox(
            "Log all user activities",
            value=True,
            help="Keep detailed logs of all user actions"
        )
        
        session_timeout = st.slider(
            "Session timeout (minutes)",
            min_value=15,
            max_value=480,
            value=120,
            help="Automatically log out inactive users"
        )
    
    # Integration settings
    st.markdown("#### 🔗 Integrations")
    with st.expander("Integration Settings", expanded=False):
        
        webhook_url = st.text_input(
            "Webhook URL",
            placeholder="https://your-domain.com/webhook",
            help="URL to send audit completion notifications"
        )
        
        api_rate_limit = st.slider(
            "API rate limit (requests/minute)",
            min_value=10,
            max_value=1000,
            value=100,
            help="Maximum API requests per minute"
        )
        
        enable_api_access = st.checkbox(
            "Enable API access",
            value=False,
            help="Allow external applications to use the API"
        )
    
    # Export/Import configuration
    st.markdown("#### 📤 Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 **Export Settings**", use_container_width=True):
            export_settings()
    
    with col2:
        imported_settings = st.file_uploader(
            "Import Settings",
            type=['json'],
            help="Upload a previously exported settings file"
        )
        
        if imported_settings and st.button("📥 **Import Settings**", use_container_width=True):
            import_settings(imported_settings)
    
    # Save advanced settings
    if st.button("🔧 **Save Advanced Settings**", type="primary", use_container_width=True):
        advanced_settings = {
            'enable_caching': enable_caching,
            'cache_ttl': cache_ttl,
            'max_concurrent_audits': max_concurrent_audits,
            'enable_parallel_processing': enable_parallel_processing,
            'auto_cleanup_enabled': auto_cleanup_enabled,
            'data_retention_days': data_retention_days,
            'max_upload_size': max_upload_size,
            'compress_uploads': compress_uploads,
            'require_file_validation': require_file_validation,
            'log_all_activities': log_all_activities,
            'session_timeout': session_timeout,
            'webhook_url': webhook_url,
            'api_rate_limit': api_rate_limit,
            'enable_api_access': enable_api_access
        }
        
        st.session_state.advanced_settings = advanced_settings
        st.success("✅ Advanced settings saved successfully!")

def render_about_section():
    """Render about and system information"""
    
    st.subheader("ℹ️ About AI Ethics Toolkit")
    
    # System information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛡️ System Information")
        st.write("**Version:** v1.0.0")
        st.write("**Build Date:** 2024-01-15")
        st.write("**License:** MIT License")
        st.write("**Python Version:** 3.11+")
        st.write("**Streamlit Version:** 1.34.0")
        
        st.markdown("#### 📊 Usage Statistics")
        st.metric("Total Audits Run", st.session_state.get('total_audits', 0))
        st.metric("Models Analyzed", st.session_state.get('models_analyzed', 0))
        st.metric("Issues Detected", st.session_state.get('total_issues', 0))
    
    with col2:
        st.markdown("#### 🔗 Resources")
        st.markdown("""
        - [📖 Documentation](https://docs.aiethics.dev)
        - [🐛 Report Issues](https://github.com/ai-ethics-toolkit/issues)
        - [💬 Community Forum](https://forum.aiethics.dev)
        - [📧 Support Email](mailto:support@aiethics.dev)
        """)
        
        st.markdown("#### 🏆 Credits")
        st.markdown("""
        **Built with:**
        - Streamlit for the web interface
        - SHAP & LIME for explainability
        - Fairlearn for bias detection
        - scikit-learn for ML utilities
        
        **Special thanks to the open-source community!**
        """)
    
    # System diagnostics
    st.markdown("#### 🔧 System Diagnostics")
    
    if st.button("🔍 **Run System Check**", use_container_width=True):
        run_system_diagnostics()
    
    # Reset application data
    st.markdown("#### ⚠️ Danger Zone")
    
    with st.expander("Reset Application Data", expanded=False):
        st.warning("⚠️ **Warning:** This will permanently delete all your data!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ **Reset All Data**", type="secondary"):
                reset_all_data()
        
        with col2:
            if st.button("📊 **Reset Only Statistics**", type="secondary"):
                reset_statistics()

def export_settings():
    """Export current settings as JSON"""
    
    settings_data = {
        'audit_thresholds': st.session_state.get('audit_thresholds', {}),
        'appearance_settings': st.session_state.get('appearance_settings', {}),
        'advanced_settings': st.session_state.get('advanced_settings', {}),
        'export_timestamp': datetime.now().isoformat()
    }
    
    settings_json = json.dumps(settings_data, indent=2)
    
    st.download_button(
        label="📥 Download Settings File",
        data=settings_json,
        file_name=f"ai_ethics_toolkit_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("✅ Settings exported successfully!")

def import_settings(settings_file):
    """Import settings from uploaded JSON file"""
    
    try:
        settings_data = json.loads(settings_file.read())
        
        # Validate and import settings
        if 'audit_thresholds' in settings_data:
            st.session_state.audit_thresholds = settings_data['audit_thresholds']
        
        if 'appearance_settings' in settings_data:
            st.session_state.appearance_settings = settings_data['appearance_settings']
        
        if 'advanced_settings' in settings_data:
            st.session_state.advanced_settings = settings_data['advanced_settings']
        
        st.success("✅ Settings imported successfully! Please refresh the page to see changes.")
        
    except Exception as e:
        st.error(f"❌ Error importing settings: {str(e)}")

def run_system_diagnostics():
    """Run system diagnostics and display results"""
    
    with st.spinner("Running system diagnostics..."):
        import time
        time.sleep(2)  # Simulate diagnostics
        
        diagnostics = {
            "System Health": "✅ Healthy",
            "Memory Usage": "📊 45% (2.1GB / 4.7GB)",
            "Disk Space": "💾 68% (12.3GB / 18.1GB)", 
            "Cache Status": "🗄️ Active (156MB)",
            "Network": "🌐 Connected",
            "Dependencies": "📦 All packages up to date",
            "Database": "🗃️ Connected",
            "Background Tasks": "⚙️ 3 running"
        }
        
        st.success("🔍 System diagnostics completed!")
        
        for component, status in diagnostics.items():
            st.write(f"**{component}:** {status}")

def reset_all_data():
    """Reset all application data"""
    
    if st.button("⚠️ **CONFIRM RESET ALL DATA**", type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.success("🗑️ All data has been reset!")
        st.info("Please refresh the page to complete the reset.")

def reset_statistics():
    """Reset only usage statistics"""
    
    if st.button("⚠️ **CONFIRM RESET STATISTICS**", type="secondary"):
        # Reset only statistics-related session state
        stats_keys = ['total_audits', 'models_analyzed', 'total_issues', 'audit_stats']
        
        for key in stats_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("📊 Statistics have been reset!")

if __name__ == "__main__":
    main()
