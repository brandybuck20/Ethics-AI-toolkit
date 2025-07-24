import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ExplainabilityVisualizer:
    """Create visualizations for explainability analysis results"""
    
    def __init__(self):
        self.color_palette = {
            'positive': '#10B981',  # Green for positive contributions
            'negative': '#EF4444',  # Red for negative contributions
            'neutral': '#6B7280',   # Gray for neutral
            'primary': '#2E5AAC',   # Blue primary
            'secondary': '#64748B'  # Gray secondary
        }
    
    def create_feature_importance_plot(self,
                                     features: List[str],
                                     importance_values: List[float],
                                     title: str = "Feature Importance",
                                     max_features: int = 15) -> go.Figure:
        """Create horizontal bar chart for feature importance"""
        
        # Limit to top features
        if len(features) > max_features:
            # Sort by absolute importance and take top features
            sorted_data = sorted(zip(features, importance_values), 
                               key=lambda x: abs(x[1]), reverse=True)
            features, importance_values = zip(*sorted_data[:max_features])
            features, importance_values = list(features), list(importance_values)
        
        # Determine colors based on positive/negative values
        colors = [self.color_palette['positive'] if val >= 0 else self.color_palette['negative'] 
                 for val in importance_values]
        
        fig = go.Figure(go.Bar(
            x=importance_values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{val:.3f}" for val in importance_values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=max(400, len(features) * 25),
            template='plotly_white',
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def create_shap_summary_plot(self,
                               shap_data: pd.DataFrame,
                               title: str = "SHAP Summary Plot") -> go.Figure:
        """Create SHAP summary plot (beeswarm style)"""
        
        if shap_data.empty:
            return self._create_empty_plot("No SHAP data available")
        
        # Group by feature and create violin plots
        features = shap_data['feature'].unique()
        
        fig = go.Figure()
        
        for i, feature in enumerate(features[:15]):  # Limit to top 15
            feature_data = shap_data[shap_data['feature'] == feature]
            
            if len(feature_data) == 0:
                continue
            
            # Create violin plot for this feature
            fig.add_trace(go.Violin(
                y=[feature] * len(feature_data),
                x=feature_data['shap_value'],
                name=feature,
                line_color=self.color_palette['primary'],
                fillcolor=f"rgba(46, 90, 172, 0.3)",
                points='all',
                pointpos=0,
                jitter=0.3,
                showlegend=False,
                hovertemplate=f'<b>{feature}</b><br>SHAP value: %{{x:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="SHAP Value (impact on model output)",
            yaxis_title="Features",
            height=max(400, len(features) * 30),
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_waterfall_plot(self,
                            waterfall_data: Dict[str, Any],
                            title: str = "SHAP Waterfall Plot") -> go.Figure:
        """Create waterfall plot for single instance explanation"""
        
        if not waterfall_data or 'contributions' not in waterfall_data:
            return self._create_empty_plot("No waterfall data available")
        
        contributions = waterfall_data['contributions']
        base_value = waterfall_data.get('base_value', 0)
        
        # Prepare data for waterfall chart
        features = ['Base Value'] + [contrib['feature'] for contrib in contributions] + ['Prediction']
        values = [base_value]
        
        # Calculate cumulative values
        cumulative = base_value
        for contrib in contributions:
            values.append(contrib['shap_value'])
            cumulative += contrib['shap_value']
        
        values.append(cumulative)
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Base value
        fig.add_trace(go.Bar(
            x=[features[0]],
            y=[values[0]],
            name='Base Value',
            marker_color=self.color_palette['neutral'],
            text=[f"{values[0]:.3f}"],
            textposition='auto'
        ))
        
        # Contributions
        y_pos = base_value
        for i, (feature, value) in enumerate(zip(features[1:-1], values[1:-1]), 1):
            color = self.color_palette['positive'] if value >= 0 else self.color_palette['negative']
            
            fig.add_trace(go.Bar(
                x=[feature],
                y=[value],
                base=y_pos if value >= 0 else y_pos + value,
                name=f'{feature}: {value:.3f}',
                marker_color=color,
                text=[f"{value:+.3f}"],
                textposition='auto',
                showlegend=False
            ))
            
            y_pos += value
        
        # Final prediction
        fig.add_trace(go.Bar(
            x=[features[-1]],
            y=[values[-1]],
            name='Final Prediction',
            marker_color=self.color_palette['primary'],
            text=[f"{values[-1]:.3f}"],
            textposition='auto',
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Components",
            yaxis_title="Value",
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_dependence_plot(self,
                             dependence_data: Dict[str, Any],
                             title: str = None) -> go.Figure:
        """Create SHAP dependence plot"""
        
        if not dependence_data or 'feature_values' not in dependence_data:
            return self._create_empty_plot("No dependence data available")
        
        feature_name = dependence_data['feature']
        title = title or f"SHAP Dependence Plot: {feature_name}"
        
        feature_values = dependence_data['feature_values']
        shap_values = dependence_data['shap_values']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=shap_values,
            mode='markers',
            marker=dict(
                size=6,
                color=feature_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"{feature_name} Value")
            ),
            name='Data Points',
            hovertemplate=f'<b>{feature_name}</b>: %{{x}}<br>SHAP Value: %{{y:.3f}}<extra></extra>'
        ))
        
        # Add trend line if enough points
        if len(feature_values) > 5:
            # Simple trend line using numpy
            z = np.polyfit(feature_values, shap_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(feature_values), max(feature_values), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend',
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"{feature_name} Value",
            yaxis_title=f"SHAP Value for {feature_name}",
            height=500,
            template='plotly_white'
        )
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def create_lime_explanation_plot(self,
                                   lime_explanation: Dict[str, Any],
                                   title: str = "LIME Local Explanation") -> go.Figure:
        """Create bar chart for LIME explanation"""
        
        if 'feature_importance' not in lime_explanation:
            return self._create_empty_plot("No LIME explanation data available")
        
        features = lime_explanation['feature_importance']['features']
        values = lime_explanation['feature_importance']['values']
        
        # Limit to top features
        if len(features) > 10:
            # Sort by absolute importance
            sorted_data = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
            features, values = zip(*sorted_data[:10])
            features, values = list(features), list(values)
        
        # Determine colors
        colors = [self.color_palette['positive'] if val >= 0 else self.color_palette['negative'] 
                 for val in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{val:.3f}" for val in values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Feature Contribution",
            yaxis_title="Features",
            height=max(400, len(features) * 30),
            template='plotly_white',
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_explanation_quality_plot(self,
                                      explanations: List[Dict[str, Any]],
                                      title: str = "Explanation Quality Distribution") -> go.Figure:
        """Create plot showing explanation quality metrics"""
        
        if not explanations:
            return self._create_empty_plot("No explanation quality data available")
        
        # Extract quality metrics
        quality_scores = []
        instance_ids = []
        
        for i, exp in enumerate(explanations):
            if 'explanation_quality' in exp:
                quality = exp['explanation_quality']
                quality_scores.append(quality.get('quality_score', 0))
                instance_ids.append(f"Instance {i+1}")
        
        if not quality_scores:
            return self._create_empty_plot("No quality scores available")
        
        # Create histogram and scatter plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Quality Score Distribution', 'Quality by Instance'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=quality_scores,
                nbinsx=10,
                marker_color=self.color_palette['primary'],
                name='Distribution'
            ),
            row=1, col=1
        )
        
        # Scatter plot
        colors = [self.color_palette['positive'] if score >= 0.7 else 
                 self.color_palette['negative'] if score < 0.5 else 
                 self.color_palette['neutral'] for score in quality_scores]
        
        fig.add_trace(
            go.Scatter(
                x=instance_ids,
                y=quality_scores,
                mode='markers',
                marker=dict(size=10, color=colors),
                name='Instance Quality',
                hovertemplate='<b>%{x}</b><br>Quality Score: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Quality Score", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Instance", row=1, col=2)
        fig.update_yaxes(title_text="Quality Score", row=1, col=2)
        
        return fig
    
    def create_feature_interaction_heatmap(self,
                                         interactions: List[Dict[str, Any]],
                                         title: str = "Feature Interactions Heatmap") -> go.Figure:
        """Create heatmap for feature interactions"""
        
        if not interactions:
            return self._create_empty_plot("No interaction data available")
        
        # Extract unique features
        features = set()
        for interaction in interactions:
            features.add(interaction['feature_1'])
            features.add(interaction['feature_2'])
        
        features = sorted(list(features))
        n_features = len(features)
        
        # Create interaction matrix
        interaction_matrix = np.zeros((n_features, n_features))
        
        for interaction in interactions:
            f1_idx = features.index(interaction['feature_1'])
            f2_idx = features.index(interaction['feature_2'])
            strength = interaction['interaction_strength']
            
            interaction_matrix[f1_idx, f2_idx] = strength
            interaction_matrix[f2_idx, f1_idx] = strength  # Symmetric
        
        fig = go.Figure(data=go.Heatmap(
            z=interaction_matrix,
            x=features,
            y=features,
            colorscale='Viridis',
            hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>Interaction Strength: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Interaction Strength")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            height=max(400, n_features * 30),
            template='plotly_white'
        )
        
        return fig
    
    def create_explanation_comparison_plot(self,
                                         explanations: List[Dict[str, Any]],
                                         comparison_metric: str = 'shap_values',
                                         title: str = "Explanation Comparison") -> go.Figure:
        """Create plot comparing explanations across multiple instances"""
        
        if len(explanations) < 2:
            return self._create_empty_plot("Need at least 2 explanations for comparison")
        
        fig = go.Figure()
        
        # Extract common features across all explanations
        all_features = set()
        for exp in explanations:
            if 'feature_contributions' in exp:
                all_features.update(exp['feature_contributions']['features'])
        
        common_features = list(all_features)[:10]  # Limit to top 10
        
        # Plot each explanation
        for i, exp in enumerate(explanations[:5]):  # Limit to 5 instances
            feature_contribs = exp.get('feature_contributions', {})
            features = feature_contribs.get('features', [])
            
            if comparison_metric == 'shap_values' and 'shap_values' in feature_contribs:
                values = feature_contribs['shap_values']
            else:
                values = feature_contribs.get('values', [])
            
            # Create mapping for common features
            feature_values = []
            for feature in common_features:
                if feature in features:
                    idx = features.index(feature)
                    feature_values.append(values[idx] if idx < len(values) else 0)
                else:
                    feature_values.append(0)
            
            fig.add_trace(go.Scatter(
                x=common_features,
                y=feature_values,
                mode='markers+lines',
                name=f"Instance {exp.get('instance_index', i+1)}",
                marker=dict(size=8),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Contribution Values",
            height=500,
            template='plotly_white',
            xaxis={'tickangle': 45}
        )
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def create_model_complexity_plot(self,
                                   model_info: Dict[str, Any],
                                   feature_importance: List[float],
                                   title: str = "Model Complexity Analysis") -> go.Figure:
        """Create plot analyzing model complexity and interpretability"""
        
        # Create subplots for different complexity metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance Distribution', 'Feature Concentration',
                          'Model Parameters', 'Interpretability Score'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Feature importance distribution
        if feature_importance:
            fig.add_trace(
                go.Histogram(
                    x=feature_importance,
                    nbinsx=15,
                    marker_color=self.color_palette['primary'],
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Feature concentration (top features vs others)
        if feature_importance:
            sorted_importance = sorted(feature_importance, reverse=True)
            top_5_sum = sum(sorted_importance[:5])
            others_sum = sum(sorted_importance[5:])
            
            fig.add_trace(
                go.Bar(
                    x=['Top 5 Features', 'Other Features'],
                    y=[top_5_sum, others_sum],
                    marker_color=[self.color_palette['positive'], self.color_palette['neutral']],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Model parameters
        param_names = []
        param_values = []
        
        if 'n_estimators' in model_info:
            param_names.append('N Estimators')
            param_values.append(model_info['n_estimators'])
        
        if 'max_depth' in model_info:
            param_names.append('Max Depth')
            param_values.append(model_info['max_depth'])
        
        if 'n_features' in model_info:
            param_names.append('N Features')
            param_values.append(model_info['n_features'])
        
        if param_names:
            fig.add_trace(
                go.Bar(
                    x=param_names,
                    y=param_values,
                    marker_color=self.color_palette['secondary'],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Interpretability score (gauge)
        interpretability_score = self._calculate_interpretability_score(model_info, feature_importance)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=interpretability_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Interpretability"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 5], 'color': self.color_palette['negative']},
                        {'range': [5, 7], 'color': self.color_palette['neutral']},
                        {'range': [7, 10], 'color': self.color_palette['positive']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 8
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _calculate_interpretability_score(self,
                                        model_info: Dict[str, Any],
                                        feature_importance: List[float]) -> float:
        """Calculate interpretability score based on model characteristics"""
        
        score = 5.0  # Base score
        
        # Model type bonus/penalty
        model_type = model_info.get('model_type', '').lower()
        if any(interpretable in model_type for interpretable in ['linear', 'tree', 'decision']):
            score += 2.0
        elif any(complex_type in model_type for complex_type in ['neural', 'ensemble', 'forest']):
            score -= 1.0
        
        # Feature importance clarity
        if feature_importance:
            # Higher variance in importance = more interpretable
            importance_std = np.std(feature_importance)
            if importance_std > 0.1:
                score += 1.0
            
            # Fewer dominant features = more interpretable
            sorted_importance = sorted(feature_importance, reverse=True)
            if len(sorted_importance) > 5:
                top_5_ratio = sum(sorted_importance[:5]) / sum(sorted_importance)
                if top_5_ratio > 0.8:
                    score += 1.0
        
        # Model complexity penalty
        if 'n_estimators' in model_info and model_info['n_estimators'] > 100:
            score -= 0.5
        
        if 'max_depth' in model_info and model_info['max_depth'] > 10:
            score -= 0.5
        
        return max(0, min(10, score))
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message"""
        
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            template='plotly_white',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return fig
