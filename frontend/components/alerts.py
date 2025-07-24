import streamlit as st
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

def render_success_alert(
    message: str,
    title: Optional[str] = None,
    dismissible: bool = True,
    auto_dismiss: bool = False,
    auto_dismiss_delay: int = 5
):
    """Render success alert"""
    
    render_alert(
        message=message,
        title=title,
        alert_type="success",
        dismissible=dismissible,
        auto_dismiss=auto_dismiss,
        auto_dismiss_delay=auto_dismiss_delay
    )

def render_error_alert(
    message: str,
    title: Optional[str] = None,
    dismissible: bool = True,
    show_details: bool = False,
    details: Optional[str] = None
):
    """Render error alert"""
    
    render_alert(
        message=message,
        title=title,
        alert_type="error",
        dismissible=dismissible,
        show_details=show_details,
        details=details
    )

def render_warning_alert(
    message: str,
    title: Optional[str] = None,
    dismissible: bool = True,
    actions: Optional[List[Dict[str, Any]]] = None
):
    """Render warning alert"""
    
    render_alert(
        message=message,
        title=title,
        alert_type="warning",
        dismissible=dismissible,
        actions=actions
    )

def render_info_alert(
    message: str,
    title: Optional[str] = None,
    dismissible: bool = True,
    icon: Optional[str] = None
):
    """Render info alert"""
    
    render_alert(
        message=message,
        title=title,
        alert_type="info",
        dismissible=dismissible,
        icon=icon
    )

def render_alert(
    message: str,
    title: Optional[str] = None,
    alert_type: str = "info",
    dismissible: bool = True,
    auto_dismiss: bool = False,
    auto_dismiss_delay: int = 5,
    show_details: bool = False,
    details: Optional[str] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    icon: Optional[str] = None
):
    """Render customizable alert component"""
    
    # Alert configurations
    alert_configs = {
        "success": {
            "color": "#10B981",
            "bg_color": "#DCFCE7",
            "border_color": "#10B981",
            "icon": "✅"
        },
        "error": {
            "color": "#EF4444",
            "bg_color": "#FEE2E2",
            "border_color": "#EF4444",
            "icon": "❌"
        },
        "warning": {
            "color": "#F59E0B",
            "bg_color": "#FEF3C7",
            "border_color": "#F59E0B",
            "icon": "⚠️"
        },
        "info": {
            "color": "#3B82F6",
            "bg_color": "#DBEAFE",
            "border_color": "#3B82F6",
            "icon": "ℹ️"
        }
    }
    
    config = alert_configs.get(alert_type, alert_configs["info"])
    alert_icon = icon if icon else config["icon"]
    
    # Generate unique key for dismissible alerts
    alert_key = f"alert_{hash(message)}_{alert_type}"
    
    # Check if alert was dismissed
    if dismissible and st.session_state.get(f"dismissed_{alert_key}", False):
        return
    
    # Auto dismiss logic
    if auto_dismiss:
        dismiss_time_key = f"dismiss_time_{alert_key}"
        if dismiss_time_key not in st.session_state:
            st.session_state[dismiss_time_key] = datetime.now() + timedelta(seconds=auto_dismiss_delay)
        
        if datetime.now() > st.session_state[dismiss_time_key]:
            st.session_state[f"dismissed_{alert_key}"] = True
            return
    
    # Title HTML
    title_html = f'<div style="font-weight: 600; font-size: 1rem; margin-bottom: 4px;">{title}</div>' if title else ""
    
    # Message HTML
    message_html = f'<div style="margin-bottom: 8px;">{message}</div>'
    
    # Details HTML
    details_html = ""
    if show_details and details:
        details_html = f'''
        <details style="margin-top: 8px;">
            <summary style="cursor: pointer; font-weight: 500;">Show Details</summary>
            <div style="margin-top: 8px; padding: 8px; background-color: rgba(0,0,0,0.05); border-radius: 4px; font-family: monospace; font-size: 0.875rem;">
                {details}
            </div>
        </details>
        '''
    
    # Actions HTML
    actions_html = ""
    if actions:
        actions_html = '<div style="margin-top: 12px; display: flex; gap: 8px;">'
        for action in actions:
            action_html = f'''
            <button onclick="alert('Action: {action.get("label", "Action")}')" 
                    style="padding: 6px 12px; background-color: {config["color"]}; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.875rem;">
                {action.get("label", "Action")}
            </button>
            '''
            actions_html += action_html
        actions_html += '</div>'
    
    # Dismiss button HTML
    dismiss_html = ""
    if dismissible:
        dismiss_html = f'''
        <button onclick="document.getElementById('{alert_key}').style.display='none';" 
                style="float: right; background: none; border: none; font-size: 1.2rem; cursor: pointer; color: {config["color"]}; margin-left: 8px;">
            ×
        </button>
        '''
    
    # Complete alert HTML
    alert_html = f'''
    <div id="{alert_key}" style="
        background-color: {config["bg_color"]};
        border-left: 4px solid {config["border_color"]};
        color: {config["color"]};
        padding: 16px;
        margin: 16px 0;
        border-radius: 4px;
        position: relative;
        font-size: 0.875rem;
        line-height: 1.5;
    ">
        <div style="display: flex; align-items: flex-start;">
            <div style="margin-right: 12px; font-size: 1.2rem;">{alert_icon}</div>
            <div style="flex: 1;">
                {title_html}
                {message_html}
                {details_html}
                {actions_html}
            </div>
            {dismiss_html}
        </div>
    </div>
    '''
    
    st.markdown(alert_html, unsafe_allow_html=True)

def render_notification_toast(
    message: str,
    toast_type: str = "info",
    duration: int = 3000
):
    """Render toast notification (requires Streamlit 1.28+)"""
    
    try:
        # Use Streamlit's built-in toast if available
        if hasattr(st, 'toast'):
            if toast_type == "success":
                st.toast(message, icon="✅")
            elif toast_type == "error":
                st.toast(message, icon="❌")
            elif toast_type == "warning":
                st.toast(message, icon="⚠️")
            else:
                st.toast(message, icon="ℹ️")
        else:
            # Fallback to regular alert
            render_alert(message, alert_type=toast_type, auto_dismiss=True, auto_dismiss_delay=duration//1000)
    except:
        # Fallback to regular alert
        render_alert(message, alert_type=toast_type, auto_dismiss=True, auto_dismiss_delay=duration//1000)

def render_progress_alert(
    message: str,
    progress: float,
    total: float = 100,
    alert_type: str = "info"
):
    """Render alert with progress bar"""
    
    percentage = (progress / total) * 100 if total > 0 else 0
    
    alert_configs = {
        "success": {"color": "#10B981", "bg_color": "#DCFCE7"},
        "error": {"color": "#EF4444", "bg_color": "#FEE2E2"},
        "warning": {"color": "#F59E0B", "bg_color": "#FEF3C7"},
        "info": {"color": "#3B82F6", "bg_color": "#DBEAFE"}
    }
    
    config = alert_configs.get(alert_type, alert_configs["info"])
    
    progress_html = f'''
    <div style="
        background-color: {config["bg_color"]};
        border-left: 4px solid {config["color"]};
        padding: 16px;
        margin: 16px 0;
        border-radius: 4px;
    ">
        <div style="margin-bottom: 8px; font-weight: 500;">{message}</div>
        <div style="background-color: rgba(255,255,255,0.5); border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="
                background-color: {config["color"]};
                height: 100%;
                width: {percentage}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="text-align: right; font-size: 0.75rem; margin-top: 4px; opacity: 0.8;">
            {progress:.1f} / {total} ({percentage:.1f}%)
        </div>
    </div>
    '''
    
    st.markdown(progress_html, unsafe_allow_html=True)

def render_alert_banner(
    message: str,
    alert_type: str = "info",
    closable: bool = True,
    sticky: bool = False
):
    """Render full-width alert banner"""
    
    alert_configs = {
        "success": {"color": "#10B981", "bg_color": "#DCFCE7"},
        "error": {"color": "#EF4444", "bg_color": "#FEE2E2"},
        "warning": {"color": "#F59E0B", "bg_color": "#FEF3C7"},
        "info": {"color": "#3B82F6", "bg_color": "#DBEAFE"}
    }
    
    config = alert_configs.get(alert_type, alert_configs["info"])
    banner_key = f"banner_{hash(message)}_{alert_type}"
    
    # Check if banner was closed
    if closable and st.session_state.get(f"closed_{banner_key}", False):
        return
    
    # Close button HTML
    close_html = ""
    if closable:
        close_html = f'''
        <button onclick="document.getElementById('{banner_key}').style.display='none';" 
                style="background: none; border: none; font-size: 1.5rem; cursor: pointer; color: {config["color"]}; float: right;">
            ×
        </button>
        '''
    
    # Sticky positioning
    position_style = "position: sticky; top: 0; z-index: 1000;" if sticky else ""
    
    banner_html = f'''
    <div id="{banner_key}" style="
        {position_style}
        background-color: {config["bg_color"]};
        color: {config["color"]};
        padding: 12px 20px;
        margin: 0 -1rem 16px -1rem;
        text-align: center;
        font-weight: 500;
        border-bottom: 1px solid {config["color"]};
    ">
        {close_html}
        {message}
    </div>
    '''
    
    st.markdown(banner_html, unsafe_allow_html=True)

def clear_all_alerts():
    """Clear all dismissed alerts from session state"""
    
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith(('dismissed_', 'dismiss_time_', 'closed_'))]
    
    for key in keys_to_remove:
        del st.session_state[key]

def render_system_alerts():
    """Render system-level alerts based on current state"""
    
    # Check for system issues
    audit_stats = st.session_state.get('audit_stats', {})
    
    # High error rate alert
    error_rate = audit_stats.get('error_rate', 0)
    if error_rate > 0.1:  # 10% error rate
        render_warning_alert(
            f"High error rate detected: {error_rate*100:.1f}%",
            title="System Warning",
            actions=[
                {"label": "View Logs", "action": "view_logs"},
                {"label": "Contact Support", "action": "contact_support"}
            ]
        )
    
    # Low ethics score alert
    avg_score = audit_stats.get('avg_ethics_score', 10)
    if avg_score < 6:
        render_error_alert(
            f"Average ethics score is critically low: {avg_score:.1f}/10",
            title="Ethics Alert",
            show_details=True,
            details="Recent audits have identified significant ethical concerns that require immediate attention."
        )
