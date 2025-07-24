import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

def render_bias_comparison_chart(
    data: Dict[str, Any],
    title: str = "Bias Analysis Results",
    height: int = 400
):
    """Render bias comparison chart"""
    
    if not data or 'groups' not in data:
        st.warning("No bias data available for visualization")
        return
    
    # Extract data
    groups = data['groups']
    positive_rates = data.get('positive_rates', [])
    
    if not positive_rates:
        st.warning("No positive rates data available")
        return
    
    # Create bar chart
    colors = ['#EF4444' if rate == max(positive_rates) or rate == min(positive_rates) 
              else '#2E5AAC' for rate in positive_rates]
    
    fig = go.Figure(data=[
        go.Bar(
            x=groups,
            y=positive_rates,
            marker_color=colors,
            text=[f"{rate:.3f}" for rate in positive_rates],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Positive Rate: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Groups",
        yaxis_title="Positive Prediction Rate",
        height=height,
        template='plotly_white',
        showlegend=False
    )
    
    # Add threshold line if provided
    if 'threshold' in data:
        fig.add_hline(
            y=data['threshold'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold: {data['threshold']:.3f}"
        )
    
    st.plotly_chart(fig, use_container_width=True)

def render_fairness_metrics_radar(
    metrics: Dict[str, float],
    title: str = "Fairness Metrics Overview",
    height: int = 400
):
    """Render radar chart for fairness metrics"""
    
    if not metrics:
        st.warning("No fairness metrics available")
        return
    
    # Prepare data
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Fairness Score',
        line_color='#2E5AAC',
        fillcolor='rgba(46, 90, 172, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title=title,
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_privacy_risk_heatmap(
    risk_data: pd.DataFrame,
    title: str = "Privacy Risk Heatmap",
    height: int = 400
):
    """Render privacy risk heatmap"""
    
    if risk_data.empty:
        st.warning("No privacy risk data available")
        return
    
    fig = px.imshow(
        risk_data,
        title=title,
        color_continuous_scale='Reds',
        aspect='auto'
    )
    
    fig.update_layout(
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance_chart(
    features: List[str],
    importance: List[float],
    title: str = "Feature Importance",
    height: int = 500,
    max_features: int = 20
):
    """Render feature importance chart"""
    
    if not features or not importance:
        st.warning("No feature importance data available")
        return
    
    # Limit to top features
    if len(features) > max_features:
        # Sort by importance and take top features
        sorted_data = sorted(zip(features, importance), key=lambda x: abs(x[1]), reverse=True)
        features, importance = zip(*sorted_data[:max_features])
        features, importance = list(features), list(importance)
    
    # Create horizontal bar chart
    colors = ['#EF4444' if imp < 0 else '#2E5AAC' for imp in importance]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{imp:.3f}" for imp in importance],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=height,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_hallucination_distribution(
    hallucination_types: List[str],
    counts: List[int],
    title: str = "Hallucination Types Distribution",
    height: int = 400
):
    """Render pie chart for hallucination types"""
    
    if not hallucination_types or not counts:
        st.info("No hallucinations detected")
        return
    
    fig = px.pie(
        values=counts,
        names=hallucination_types,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_trend_chart(
    dates: List[str],
    values: List[float],
    metric_name: str = "Ethics Score",
    title: str = "Trend Analysis",
    height: int = 400,
    target_line: Optional[float] = None
):
    """Render trend line chart"""
    
    if not dates or not values:
        st.warning("No trend data available")
        return
    
    fig = go.Figure()
    
    # Main trend line
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name=metric_name,
        line=dict(color='#2E5AAC', width=3),
        marker=dict(size=6),
        hovertemplate=f'<b>Date:</b> %{{x}}<br><b>{metric_name}:</b> %{{y:.2f}}<extra></extra>'
    ))
    
    # Add target line if provided
    if target_line is not None:
        fig.add_hline(
            y=target_line,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Target: {target_line}"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=metric_name,
        height=height,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_confidence_distribution(
    confidence_scores: List[float],
    title: str = "Confidence Score Distribution",
    height: int = 400
):
    """Render histogram of confidence scores"""
    
    if not confidence_scores:
        st.warning("No confidence data available")
        return
    
    fig = px.histogram(
        x=confidence_scores,
        nbins=20,
        title=title,
        labels={'x': 'Confidence Score', 'y': 'Count'},
        color_discrete_sequence=['#2E5AAC']
    )
    
    fig.update_layout(
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_comparison_chart(
    categories: List[str],
    series_data: Dict[str, List[float]],
    title: str = "Comparison Chart",
    height: int = 400,
    chart_type: str = "bar"  # 'bar' or 'line'
):
    """Render comparison chart with multiple series"""
    
    if not categories or not series_data:
        st.warning("No comparison data available")
        return
    
    fig = go.Figure()
    
    colors = ['#2E5AAC', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6']
    
    for i, (series_name, values) in enumerate(series_data.items()):
        if chart_type == "bar":
            fig.add_trace(go.Bar(
                name=series_name,
                x=categories,
                y=values,
                marker_color=colors[i % len(colors)]
            ))
        else:  # line chart
            fig.add_trace(go.Scatter(
                name=series_name,
                x=categories,
                y=values,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        barmode='group' if chart_type == "bar" else None
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_gauge_chart(
    value: float,
    title: str = "Score",
    min_val: float = 0,
    max_val: float = 10,
    threshold_good: float = 8,
    threshold_warning: float = 6,
    height: int = 300
):
    """Render gauge chart for scores"""
    
    # Determine color based on thresholds
    if value >= threshold_good:
        color = "#10B981"  # Green
    elif value >= threshold_warning:
        color = "#F59E0B"  # Yellow
    else:
        color = "#EF4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': threshold_good},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, threshold_warning], 'color': "lightgray"},
                {'range': [threshold_warning, threshold_good], 'color': "gray"},
                {'range': [threshold_good, max_val], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_matrix_chart(
    matrix_data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str = "Confusion Matrix",
    height: int = 400,
    color_scale: str = 'Blues'
):
    """Render matrix/heatmap chart"""
    
    fig = px.imshow(
        matrix_data,
        x=x_labels,
        y=y_labels,
        title=title,
        color_continuous_scale=color_scale,
        text_auto=True
    )
    
    fig.update_layout(
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
