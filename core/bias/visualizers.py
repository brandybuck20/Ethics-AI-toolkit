import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class BiasVisualizer:
    """Create visualizations for bias analysis results"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#2E5AAC',
            'secondary': '#64748B',
            'success': '#10B981',
            'warning': '#F59E0B',
            'error': '#EF4444',
            'info': '#3B82F6'
        }
    
    def create_demographic_parity_chart(self,
                                      group_metrics: Dict[str, Dict],
                                      protected_attribute: str,
                                      threshold: float = 0.1) -> go.Figure:
        """Create demographic parity visualization"""
        
        groups = list(group_metrics.keys())
        positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
        group_sizes = [metrics['size'] for metrics in group_metrics.values()]
        
        # Determine colors based on bias
        max_rate = max(positive_rates)
        min_rate = min(positive_rates)
        bias_detected = (max_rate - min_rate) > threshold
        
        colors = []
        for rate in positive_rates:
            if bias_detected and (rate == max_rate or rate == min_rate):
                colors.append(self.color_palette['error'])
            else:
                colors.append(self.color_palette['primary'])
        
        # Create subplot with secondary y-axis for group sizes
        fig = make_subplots(
            rows=1, cols=1,
            secondary_y=True,
            subplot_titles=[f'Demographic Parity Analysis - {protected_attribute.title()}']
        )
        
        # Bar chart for positive rates
        fig.add_trace(
            go.Bar(
                x=groups,
                y=positive_rates,
                name='Positive Rate',
                marker_color=colors,
                text=[f"{rate:.3f}" for rate in positive_rates],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Positive Rate: %{y:.3f}<br>Group Size: %{customdata}<extra></extra>',
                customdata=group_sizes
            ),
            secondary_y=False
        )
        
        # Add threshold lines
        if len(positive_rates) > 1:
            fig.add_hline(
                y=max_rate - threshold,
                line_dash="dash",
                line_color=self.color_palette['warning'],
                annotation_text=f"Acceptable Range (-{threshold:.2f})",
                secondary_y=False
            )
            
            fig.add_hline(
                y=min_rate + threshold,
                line_dash="dash", 
                line_color=self.color_palette['warning'],
                annotation_text=f"Acceptable Range (+{threshold:.2f})",
                secondary_y=False
            )
        
        # Set axis labels
        fig.update_yaxes(title_text="Positive Prediction Rate", secondary_y=False)
        fig.update_xaxes(title_text=protected_attribute.title())
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            showlegend=False,
            title_x=0.5
        )
        
        return fig
    
    def create_fairness_metrics_radar(self,
                                    fairness_metrics: Dict[str, float],
                                    threshold: float = 0.1) -> go.Figure:
        """Create radar chart for multiple fairness metrics"""
        
        metrics_display = {
            'demographic_parity_difference': 'Demographic Parity',
            'equalized_odds_difference': 'Equalized Odds',
            'equal_opportunity_difference': 'Equal Opportunity',
            'calibration_difference': 'Calibration',
            'individual_fairness_score': 'Individual Fairness'
        }
        
        categories = []
        values = []
        
        for metric_key, display_name in metrics_display.items():
            if metric_key in fairness_metrics:
                categories.append(display_name)
                
                # For individual fairness, higher is better, so invert
                if metric_key == 'individual_fairness_score':
                    values.append(fairness_metrics[metric_key])
                else:
                    # For other metrics, lower is better, so invert for display
                    normalized_value = max(0, 1 - (fairness_metrics[metric_key] / threshold))
                    values.append(normalized_value)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Fairness Score',
            line_color=self.color_palette['primary'],
            fillcolor=f"rgba(46, 90, 172, 0.2)"
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    ticktext=['Poor', 'Fair', 'Good', 'Excellent'],
                    tickvals=[0.25, 0.5, 0.75, 1.0]
                )),
            title="Fairness Metrics Overview",
            title_x=0.5,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_intersectional_bias_heatmap(self,
                                         intersectional_metrics: Dict[str, Dict],
                                         attributes: List[str]) -> go.Figure:
        """Create heatmap for intersectional bias analysis"""
        
        # Parse intersectional group names
        groups = list(intersectional_metrics.keys())
        
        # Create matrix data
        attr1_values = set()
        attr2_values = set()
        
        for group in groups:
            parts = group.split('_')
            if len(parts) >= 2:
                attr1_values.add(parts[0])
                attr2_values.add(parts[1])
        
        attr1_values = sorted(list(attr1_values))
        attr2_values = sorted(list(attr2_values))
        
        # Create bias matrix
        bias_matrix = []
        hover_text = []
        
        for attr2 in attr2_values:
            row = []
            hover_row = []
            
            for attr1 in attr1_values:
                group_key = f"{attr1}_{attr2}"
                
                if group_key in intersectional_metrics:
                    positive_rate = intersectional_metrics[group_key]['positive_rate']
                    group_size = intersectional_metrics[group_key]['size']
                    row.append(positive_rate)
                    hover_row.append(f"Group: {attr1}_{attr2}<br>Positive Rate: {positive_rate:.3f}<br>Size: {group_size}")
                else:
                    row.append(0)
                    hover_row.append(f"Group: {attr1}_{attr2}<br>No data")
            
            bias_matrix.append(row)
            hover_text.append(hover_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=bias_matrix,
            x=attr1_values,
            y=attr2_values,
            colorscale='RdYlBu_r',
            hoverinfo='text',
            text=hover_text,
            colorbar=dict(title="Positive Rate")
        ))
        
        fig.update_layout(
            title=f"Intersectional Bias Heatmap - {attributes[0].title()} vs {attributes[1].title()}",
            title_x=0.5,
            xaxis_title=attributes[0].title(),
            yaxis_title=attributes[1].title(),
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_bias_trend_chart(self,
                              historical_data: List[Dict],
                              protected_attribute: str) -> go.Figure:
        """Create trend chart for bias metrics over time"""
        
        if not historical_data:
            return go.Figure().add_annotation(
                text="No historical data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Add trend line for demographic parity
        if 'demographic_parity_difference' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['demographic_parity_difference'],
                mode='lines+markers',
                name='Demographic Parity Difference',
                line=dict(color=self.color_palette['primary'], width=3),
                marker=dict(size=6)
            ))
        
        # Add threshold line
        threshold = 0.1  # Default threshold
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=self.color_palette['warning'],
            annotation_text=f"Bias Threshold ({threshold})"
        )
        
        fig.update_layout(
            title=f"Bias Trend Analysis - {protected_attribute.title()}",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Bias Metric Value",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_group_comparison_chart(self,
                                    group_metrics: Dict[str, Dict],
                                    metric_name: str = 'positive_rate') -> go.Figure:
        """Create comparison chart for different metrics across groups"""
        
        groups = list(group_metrics.keys())
        values = [metrics.get(metric_name, 0) for metrics in group_metrics.values()]
        
        # Create bar chart with error indication
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=groups,
            y=values,
            name=metric_name.replace('_', ' ').title(),
            marker_color=self.color_palette['primary'],
            text=[f"{val:.3f}" for val in values],
            textposition='auto'
        ))
        
        # Add average line
        avg_value = np.mean(values)
        fig.add_hline(
            y=avg_value,
            line_dash="dash",
            line_color=self.color_palette['secondary'],
            annotation_text=f"Average: {avg_value:.3f}"
        )
        
        fig.update_layout(
            title=f"{metric_name.replace('_', ' ').title()} by Group",
            title_x=0.5,
            xaxis_title="Groups",
            yaxis_title=metric_name.replace('_', ' ').title(),
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_privilege_gap_chart(self,
                                 privilege_gaps: Dict[str, float]) -> go.Figure:
        """Create chart showing privilege gaps between groups"""
        
        groups = list(privilege_gaps.keys())
        gaps = list(privilege_gaps.values())
        
        # Sort by gap size
        sorted_data = sorted(zip(groups, gaps), key=lambda x: x[1], reverse=True)
        groups, gaps = zip(*sorted_data)
        
        # Color code by gap size
        colors = []
        for gap in gaps:
            if gap > 0.15:
                colors.append(self.color_palette['error'])
            elif gap > 0.05:
                colors.append(self.color_palette['warning'])
            else:
                colors.append(self.color_palette['success'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(groups),
            y=list(gaps),
            name='Privilege Gap',
            marker_color=colors,
            text=[f"{gap:.3f}" for gap in gaps],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Privilege Gap: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Privilege Gaps Across Intersectional Groups",
            title_x=0.5,
            xaxis_title="Groups",
            yaxis_title="Privilege Gap",
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
