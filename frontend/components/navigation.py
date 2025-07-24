import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime

def render_main_navigation():
    """Render the main sidebar navigation"""
    
    with st.sidebar:
        # Header with logo and title
        render_navigation_header()
        
        # Main navigation menu
        selected_page = render_navigation_menu()
        
        # Quick stats section
        render_quick_stats()
        
        # Footer with system info
        render_navigation_footer()
        
        return selected_page

def render_navigation_header():
    """Render navigation header with branding"""
    
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <div style="font-size: 3em; margin-bottom: 10px;">🛡️</div>
        <h2 style="color: #2E5AAC; margin: 0; font-size: 1.4em; font-weight: 700;">
            AI Ethics Toolkit
        </h2>
        <p style="color: #64748B; margin: 5px 0 0 0; font-size: 0.85em;">
            Comprehensive AI Auditing Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

def render_navigation_menu():
    """Render the main navigation menu"""
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'Home')
    
    # Navigation options
    nav_options = [
        "Home", 
        "Bias Audit", 
        "Privacy Analysis", 
        "Explainability",
        "Hallucination Check", 
        "Audit Reports", 
        "Settings"
    ]
    
    # Navigation icons
    nav_icons = [
        "house", 
        "bar-chart-line", 
        "shield-lock", 
        "lightbulb",
        "search", 
        "file-text", 
        "gear"
    ]
    
    # Get current index
    try:
        current_index = nav_options.index(current_page)
    except ValueError:
        current_index = 0
    
    # Render option menu
    selected = option_menu(
        menu_title="Navigation",
        options=nav_options,
        icons=nav_icons,
        menu_icon="list",
        default_index=current_index,
        styles={
            "container": {
                "padding": "0!important", 
                "background-color": "transparent",
                "border-radius": "8px"
            },
            "icon": {
                "color": "#2E5AAC", 
                "font-size": "16px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "12px 16px",
                "border-radius": "8px",
                "color": "#374151",
                "font-weight": "500"
            },
            "nav-link-selected": {
                "background-color": "#2E5AAC",
                "color": "white",
                "font-weight": "600"
            },
            "nav-link:hover": {
                "background-color": "#F1F5F9",
                "color": "#2E5AAC"
            }
        }
    )
    
    # Update session state
    st.session_state.current_page = selected
    
    return selected

def render_quick_stats():
    """Render quick statistics in sidebar"""
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Get stats from session state
    stats = st.session_state.get('audit_stats', {
        'total_audits': 0,
        'issues_found': 0,
        'avg_ethics_score': 0.0,
        'last_audit_date': None
    })
    
    # Display metrics in compact format
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 8px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px;">
            <div style="font-size: 1.2em; font-weight: bold; color: #2E5AAC;">{stats['total_audits']}</div>
            <div style="font-size: 0.75em; color: #64748B;">Audits</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 8px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px;">
            <div style="font-size: 1.2em; font-weight: bold; color: #EF4444;">{stats['issues_found']}</div>
            <div style="font-size: 0.75em; color: #64748B;">Issues</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Ethics score with color coding
    score = stats['avg_ethics_score']
    score_color = "#10B981" if score >= 8 else "#F59E0B" if score >= 6 else "#EF4444"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 8px; background: #F8FAFC; border-radius: 6px; margin-bottom: 8px;">
        <div style="font-size: 1.2em; font-weight: bold; color: {score_color};">{score:.1f}/10</div>
        <div style="font-size: 0.75em; color: #64748B;">Ethics Score</div>
    </div>
    """, unsafe_allow_html=True)

def render_navigation_footer():
    """Render navigation footer"""
    
    st.markdown("---")
    
    # System status indicator
    st.markdown("""
    <div style="text-align: center; padding: 8px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%;"></div>
            <span style="font-size: 0.8em; color: #64748B;">System Online</span>
        </div>
        <div style="font-size: 0.75em; color: #9CA3AF; margin-top: 4px;">
            v1.0.0 • {datetime.now().strftime('%H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_breadcrumb(current_page: str, sub_page: str = None):
    """Render breadcrumb navigation"""
    
    if sub_page:
        breadcrumb = f"🏠 Home > {current_page} > {sub_page}"
    else:
        breadcrumb = f"🏠 Home > {current_page}"
    
    st.markdown(f"""
    <div style="padding: 8px 0; margin-bottom: 16px; font-size: 0.9em; color: #64748B;">
        {breadcrumb}
    </div>
    """, unsafe_allow_html=True)
