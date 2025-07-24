import streamlit as st
from typing import Optional, Union, List, Dict, Any
import plotly.graph_objects as go

def render_metric_card(
    title: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
    icon: Optional[str] = None,
    background_color: str = "#FFFFFF",
    border_color: str = "#E2E8F0"
):
    """Render a styled metric card"""
    
    # Format delta with appropriate color
    delta_colors = {
        "normal": "#10B981",
        "inverse": "#EF4444",
        "off": "#64748B"
    }
    
    delta_html = ""
    if delta is not None:
        color = delta_colors.get(delta_color, "#64748B")
        arrow = "↗" if str(delta).startswith("+") or (isinstance(delta, (int, float)) and delta > 0) else "↘" if str(delta).startswith("-") or (isinstance(delta, (int, float)) and delta < 0) else ""
        delta_html = f'<div style="color: {color}; font-size: 0.875rem; margin-top: 4px;">{arrow} {delta}</div>'
    
    # Add icon if provided
    icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 8px;">{icon}</div>' if icon else ""
    
    # Help text
    help_html = f'<div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">{help_text}</div>' if help_text else ""
    
    metric_html = f"""
    <div style="
        background-color: {background_color};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
        margin: 8px 0;
        transition: all 0.2s ease;
    " onmouseover="this.style.boxShadow='0 4px 6px -1px rgb(0 0 0 / 0.1)'; this.style.transform='translateY(-2px)'"
       onmouseout="this.style.boxShadow='0 1px 3px 0 rgb(0 0 0 / 0.1)'; this.style.transform='translateY(0)'">
        {icon_html}
        <div style="color: #374151; font-size: 0.875rem; font-weight: 500; margin-bottom: 4px;">{title}</div>
        <div style="color: #111827; font-size: 1.875rem; font-weight: 700; line-height: 1;">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """
    
    st.markdown(metric_html, unsafe_allow_html=True)

def render_metrics_grid(
    metrics: List[Dict[str, Any]],
    columns: int = 4
):
    """Render a grid of metric cards"""
    
    # Create columns
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            render_metric_card(**metric)

def render_progress_metric(
    title: str,
    current: Union[int, float],
    target: Union[int, float],
    unit: str = "",
    color: str = "#2E5AAC",
    height: int = 60,
    show_percentage: bool = True
):
    """Render a progress-style metric"""
    
    # Calculate progress percentage
    progress = min(current / target * 100, 100) if target > 0 else 0
    
    # Create progress bar HTML
    progress_html = f"""
    <div style="margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #374151;">{title}</span>
            <span style="font-size: 0.875rem; color: #64748B;">
                {current}{unit} / {target}{unit}
                {f" ({progress:.1f}%)" if show_percentage else ""}
            </span>
        </div>
        <div style="background-color: #F3F4F6; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="
                background-color: {color};
                height: 100%;
                width: {progress}%;
                transition: width 0.3s ease;
                border-radius: 4px;
            "></div>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

def render_status_indicator(
    status: str,
    label: str = "Status",
    show_dot: bool = True
):
    """Render status indicator with colored dot"""
    
    status_colors = {
        "online": "#10B981",
        "offline": "#EF4444", 
        "warning": "#F59E0B",
        "info": "#3B82F6",
        "success": "#10B981",
        "error": "#EF4444",
        "pending": "#F59E0B"
    }
    
    color = status_colors.get(status.lower(), "#6B7280")
    dot_html = f'<span style="color: {color}; margin-right: 8px;">●</span>' if show_dot else ""
    
    status_html = f"""
    <div style="display: flex; align-items: center; padding: 8px 0;">
        {dot_html}
        <span style="font-weight: 500; color: #374151; margin-right: 8px;">{label}:</span>
        <span style="color: {color}; font-weight: 600; text-transform: capitalize;">{status}</span>
    </div>
    """
    
    st.markdown(status_html, unsafe_allow_html=True)

def render_score_badge(
    score: Union[int, float],
    max_score: Union[int, float] = 10,
    size: str = "medium",  # "small", "medium", "large"
    show_label: bool = True
):
    """Render a score badge with color coding"""
    
    # Calculate percentage
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    
    # Determine color based on score
    if percentage >= 80:
        color = "#10B981"
        bg_color = "#DCFCE7"
    elif percentage >= 60:
        color = "#F59E0B"
        bg_color = "#FEF3C7"
    else:
        color = "#EF4444"
        bg_color = "#FEE2E2"
    
    # Size configurations
    size_configs = {
        "small": {"font_size": "0.75rem", "padding": "4px 8px"},
        "medium": {"font_size": "0.875rem", "padding": "6px 12px"},
        "large": {"font_size": "1rem", "padding": "8px 16px"}
    }
    
    config = size_configs.get(size, size_configs["medium"])
    label_text = "Score: " if show_label else ""
    
    badge_html = f"""
    <span style="
        background-color: {bg_color};
        color: {color};
        font-size: {config['font_size']};
        font-weight: 600;
        padding: {config['padding']};
        border-radius: 12px;
        display: inline-block;
        margin: 4px 0;
    ">
        {label_text}{score}/{max_score}
    </span>
    """
    
    st.markdown(badge_html, unsafe_allow_html=True)

def render_trend_indicator(
    current_value: Union[int, float],
    previous_value: Union[int, float],
    label: str = "Trend",
    show_percentage: bool = True,
    invert_colors: bool = False
):
    """Render trend indicator with arrow and percentage change"""
    
    if previous_value == 0:
        change_pct = 0
    else:
        change_pct = ((current_value - previous_value) / previous_value) * 100
    
    # Determine trend direction and color
    if change_pct > 0:
        arrow = "↗"
        color = "#EF4444" if invert_colors else "#10B981"
        trend = "up"
    elif change_pct < 0:
        arrow = "↘"
        color = "#10B981" if invert_colors else "#EF4444"
        trend = "down"
    else:
        arrow = "→"
        color = "#6B7280"
        trend = "stable"
    
    percentage_text = f" ({abs(change_pct):.1f}%)" if show_percentage and change_pct != 0 else ""
    
    trend_html = f"""
    <div style="display: flex; align-items: center; gap: 8px; padding: 4px 0;">
        <span style="font-weight: 500; color: #374151;">{label}:</span>
        <span style="color: {color}; font-weight: 600;">
            {arrow} {trend.title()}{percentage_text}
        </span>
    </div>
    """
    
    st.markdown(trend_html, unsafe_allow_html=True)

def render_metric_comparison(
    metrics: Dict[str, Union[int, float]],
    title: str = "Comparison",
    highlight_best: bool = True,
    higher_is_better: bool = True
):
    """Render comparison of multiple metrics"""
    
    if not metrics:
        st.warning("No metrics to compare")
        return
    
    st.markdown(f"**{title}**")
    
    # Find best/worst values
    if highlight_best:
        best_value = max(metrics.values()) if higher_is_better else min(metrics.values())
        worst_value = min(metrics.values()) if higher_is_better else max(metrics.values())
    
    for name, value in metrics.items():
        # Determine styling
        if highlight_best:
            if value == best_value:
                style = "color: #10B981; font-weight: 700;"
            elif value == worst_value:
                style = "color: #EF4444; font-weight: 500;"
            else:
                style = "color: #374151; font-weight: 500;"
        else:
            style = "color: #374151; font-weight: 500;"
        
        comparison_html = f"""
        <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #F3F4F6;">
            <span style="color: #64748B;">{name}</span>
            <span style="{style}">{value}</span>
        </div>
        """
        
        st.markdown(comparison_html, unsafe_allow_html=True)

def render_circular_progress(
    percentage: float,
    title: str = "Progress",
    size: int = 120,
    color: str = "#2E5AAC"
):
    """Render circular progress indicator"""
    
    # Clamp percentage between 0 and 100
    percentage = max(0, min(100, percentage))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': "%", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [{'range': [0, 100], 'color': 'lightgray'}],
        }
    ))
    
    fig.update_layout(
        height=size,
        width=size,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=False)
