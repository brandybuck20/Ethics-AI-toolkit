import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import json
import re

class DataFormatter:
    """Data formatting utilities for AI Ethics Toolkit"""
    
    @staticmethod
    def format_number(value: Union[int, float], 
                     decimal_places: int = 2,
                     use_thousands_separator: bool = True) -> str:
        """Format numbers for display"""
        
        if pd.isna(value):
            return "N/A"
        
        try:
            if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                if use_thousands_separator:
                    return f"{int(value):,}"
                else:
                    return str(int(value))
            else:
                if use_thousands_separator:
                    return f"{value:,.{decimal_places}f}"
                else:
                    return f"{value:.{decimal_places}f}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def format_percentage(value: Union[int, float], 
                         decimal_places: int = 1) -> str:
        """Format percentage values"""
        
        if pd.isna(value):
            return "N/A"
        
        try:
            return f"{value * 100:.{decimal_places}f}%"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def format_datetime(dt: Union[datetime, str, pd.Timestamp],
                       format_type: str = "default") -> str:
        """Format datetime values"""
        
        if pd.isna(dt) or dt is None:
            return "N/A"
        
        # Convert to datetime if string
        if isinstance(dt, str):
            try:
                dt = pd.to_datetime(dt)
            except:
                return str(dt)
        
        format_patterns = {
            "default": "%Y-%m-%d %H:%M:%S",
            "date_only": "%Y-%m-%d",
            "time_only": "%H:%M:%S",
            "friendly": "%B %d, %Y at %I:%M %p",
            "short": "%m/%d/%Y %H:%M",
            "iso": "%Y-%m-%dT%H:%M:%S"
        }
        
        pattern = format_patterns.get(format_type, format_patterns["default"])
        
        try:
            return dt.strftime(pattern)
        except (AttributeError, ValueError):
            return str(dt)

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"

    @staticmethod
    def format_duration(seconds: Union[int, float]) -> str:
        """Format duration in human readable format"""
        
        if pd.isna(seconds) or seconds < 0:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        else:
            days = seconds / 86400
            return f"{days:.1f} days"

    @staticmethod
    def format_score(score: Union[int, float],
                    max_score: Union[int, float] = 10,
                    show_max: bool = True) -> str:
        """Format score values"""
        
        if pd.isna(score):
            return "N/A"
        
        try:
            if show_max:
                return f"{score:.1f}/{max_score}"
            else:
                return f"{score:.1f}"
        except (ValueError, TypeError):
            return str(score)

    @staticmethod
    def format_confidence(confidence: Union[int, float]) -> str:
        """Format confidence values"""
        
        if pd.isna(confidence):
            return "N/A"
        
        try:
            # Convert to percentage if between 0 and 1
            if 0 <= confidence <= 1:
                return f"{confidence * 100:.1f}%"
            else:
                return f"{confidence:.1f}"
        except (ValueError, TypeError):
            return str(confidence)

    @staticmethod
    def truncate_text(text: str, 
                     max_length: int = 100,
                     suffix: str = "...") -> str:
        """Truncate text to specified length"""
        
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix

    @staticmethod
    def format_list(items: List[Any],
                   max_items: int = 5,
                   separator: str = ", ") -> str:
        """Format list for display"""
        
        if not items:
            return "None"
        
        # Convert items to strings
        str_items = [str(item) for item in items]
        
        if len(str_items) <= max_items:
            return separator.join(str_items)
        else:
            displayed_items = str_items[:max_items]
            remaining_count = len(str_items) - max_items
            return separator.join(displayed_items) + f" (+{remaining_count} more)"

class MetricFormatter:
    """Specialized formatters for metrics and statistics"""
    
    @staticmethod
    def format_bias_metric(value: float, 
                          threshold: float = 0.1,
                          format_type: str = "difference") -> Dict[str, Any]:
        """Format bias metrics with context"""
        
        if pd.isna(value):
            return {
                "formatted_value": "N/A",
                "status": "unknown",
                "severity": "info"
            }
        
        # Format value based on type
        if format_type == "difference":
            formatted_value = f"{value:.3f}"
        elif format_type == "ratio":
            formatted_value = f"{value:.2f}:1"
        else:
            formatted_value = f"{value:.3f}"
        
        # Determine status
        if abs(value) <= threshold:
            status = "acceptable"
            severity = "success"
        elif abs(value) <= threshold * 2:
            status = "concerning"
            severity = "warning"
        else:
            status = "problematic"
            severity = "error"
        
        return {
            "formatted_value": formatted_value,
            "status": status,
            "severity": severity,
            "threshold": threshold
        }

    @staticmethod
    def format_privacy_risk(risk_level: str,
                           confidence: float = None) -> Dict[str, Any]:
        """Format privacy risk levels"""
        
        risk_configs = {
            "low": {"color": "#10B981", "icon": "✅", "description": "Low privacy risk"},
            "medium": {"color": "#F59E0B", "icon": "⚠️", "description": "Medium privacy risk"},
            "high": {"color": "#EF4444", "icon": "🚨", "description": "High privacy risk"},
            "critical": {"color": "#DC2626", "icon": "🔴", "description": "Critical privacy risk"}
        }
        
        config = risk_configs.get(risk_level.lower(), risk_configs["medium"])
        
        result = {
            "level": risk_level.title(),
            "color": config["color"],
            "icon": config["icon"],
            "description": config["description"]
        }
        
        if confidence is not None:
            result["confidence"] = DataFormatter.format_confidence(confidence)
        
        return result

    @staticmethod
    def format_explainability_score(score: float) -> Dict[str, Any]:
        """Format explainability scores with interpretation"""
        
        if pd.isna(score):
            return {
                "score": "N/A",
                "interpretation": "Unknown",
                "color": "#6B7280"
            }
        
        # Interpret score
        if score >= 8.0:
            interpretation = "Highly Interpretable"
            color = "#10B981"
        elif score >= 6.0:
            interpretation = "Moderately Interpretable"
            color = "#F59E0B"
        elif score >= 4.0:
            interpretation = "Somewhat Interpretable"
            color = "#FF8C00"
        else:
            interpretation = "Poorly Interpretable"
            color = "#EF4444"
        
        return {
            "score": DataFormatter.format_score(score),
            "interpretation": interpretation,
            "color": color
        }

class TableFormatter:
    """Formatters for tabular data display"""
    
    @staticmethod
    def format_dataframe_for_display(df: pd.DataFrame,
                                   max_rows: int = 100,
                                   float_precision: int = 3) -> pd.DataFrame:
        """Format DataFrame for better display"""
        
        # Create a copy to avoid modifying original
        display_df = df.copy()
        
        # Limit rows
        if len(display_df) > max_rows:
            display_df = display_df.head(max_rows)
        
        # Format numeric columns
        for col in display_df.select_dtypes(include=[np.number]).columns:
            if display_df[col].dtype == 'float64':
                display_df[col] = display_df[col].round(float_precision)
        
        # Format datetime columns
        for col in display_df.select_dtypes(include=['datetime64']).columns:
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M')
        
        # Truncate long text columns
        for col in display_df.select_dtypes(include=['object']).columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str).apply(
                    lambda x: DataFormatter.truncate_text(x, 50)
                )
        
        return display_df

    @staticmethod
    def create_summary_table(data: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table from dictionary"""
        
        summary_data = []
        
        for key, value in data.items():
            # Format key (convert snake_case to Title Case)
            formatted_key = key.replace('_', ' ').title()
            
            # Format value based on type
            if isinstance(value, (int, float)):
                if isinstance(value, float) and 0 <= value <= 1:
                    formatted_value = DataFormatter.format_percentage(value)
                else:
                    formatted_value = DataFormatter.format_number(value)
            elif isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
            elif isinstance(value, list):
                formatted_value = DataFormatter.format_list(value)
            elif isinstance(value, datetime):
                formatted_value = DataFormatter.format_datetime(value)
            else:
                formatted_value = str(value)
            
            summary_data.append({
                'Metric': formatted_key,
                'Value': formatted_value
            })
        
        return pd.DataFrame(summary_data)

class ReportFormatter:
    """Formatters for generating reports"""
    
    @staticmethod
    def format_audit_report(audit_results: Dict[str, Any],
                           report_type: str = "comprehensive") -> str:
        """Format audit results into a readable report"""
        
        timestamp = DataFormatter.format_datetime(datetime.now(), "friendly")
        
        # Report header
        report = f"""
AI ETHICS TOOLKIT - AUDIT REPORT
================================

Generated: {timestamp}
Report Type: {report_type.title()}

"""
        
        # Executive Summary
        if 'summary' in audit_results:
            report += "EXECUTIVE SUMMARY\n"
            report += "-" * 17 + "\n"
            
            summary = audit_results['summary']
            for key, value in summary.items():
                formatted_key = key.replace('_', ' ').title()
                
                if isinstance(value, (int, float)):
                    if 0 <= value <= 1:
                        formatted_value = DataFormatter.format_percentage(value)
                    else:
                        formatted_value = DataFormatter.format_number(value)
                else:
                    formatted_value = str(value)
                
                report += f"{formatted_key}: {formatted_value}\n"
            
            report += "\n"
        
        # Detailed findings
        if 'findings' in audit_results and audit_results['findings']:
            report += "DETAILED FINDINGS\n"
            report += "-" * 17 + "\n"
            
            for i, finding in enumerate(audit_results['findings'], 1):
                report += f"\nFinding #{i}: {finding.get('type', 'Unknown')}\n"
                report += f"Severity: {finding.get('severity', 'Unknown')}\n"
                report += f"Description: {finding.get('description', 'No description')}\n"
                
                if 'confidence' in finding:
                    confidence = DataFormatter.format_confidence(finding['confidence'])
                    report += f"Confidence: {confidence}\n"
                
                if 'recommendation' in finding:
                    report += f"Recommendation: {finding['recommendation']}\n"
            
            report += "\n"
        
        # Recommendations
        if 'recommendations' in audit_results:
            report += "RECOMMENDATIONS\n"
            report += "-" * 15 + "\n"
            
            for i, rec in enumerate(audit_results['recommendations'], 1):
                clean_rec = rec.replace('**', '').replace('*', '')
                # Remove emoji characters
                clean_rec = re.sub(r'[^\w\s\-.,;:!?()]', '', clean_rec)
                report += f"{i}. {clean_rec}\n"
        
        return report

    @staticmethod
    def format_json_report(data: Dict[str, Any]) -> str:
        """Format data as pretty JSON"""
        
        def json_serializer(obj):
            """Custom JSON serializer for special types"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return str(obj)
        
        try:
            return json.dumps(data, indent=2, default=json_serializer, ensure_ascii=False)
        except Exception as e:
            return f"Error formatting JSON: {str(e)}"

class ChartFormatter:
    """Formatters for chart data and labels"""
    
    @staticmethod
    def format_chart_data(data: Dict[str, List],
                         chart_type: str = "bar") -> Dict[str, List]:
        """Format data for chart display"""
        
        formatted_data = {}
        
        for key, values in data.items():
            formatted_key = key.replace('_', ' ').title()
            
            if chart_type in ['bar', 'line', 'scatter']:
                # Format numeric values
                formatted_values = []
                for value in values:
                    if isinstance(value, (int, float)):
                        formatted_values.append(round(value, 3))
                    else:
                        formatted_values.append(value)
                formatted_data[formatted_key] = formatted_values
            else:
                formatted_data[formatted_key] = values
        
        return formatted_data

    @staticmethod
    def generate_chart_title(chart_type: str,
                           metrics: List[str],
                           context: str = "") -> str:
        """Generate descriptive chart titles"""
        
        metric_text = DataFormatter.format_list(metrics, max_items=3)
        
        title_templates = {
            "bar": f"{metric_text} Comparison",
            "line": f"{metric_text} Trends",
            "pie": f"{metric_text} Distribution",
            "scatter": f"{metric_text} Correlation",
            "heatmap": f"{metric_text} Heatmap",
            "histogram": f"{metric_text} Distribution"
        }
        
        base_title = title_templates.get(chart_type, f"{metric_text} Analysis")
        
        if context:
            return f"{base_title} - {context}"
        
        return base_title

# Convenience functions for common formatting needs
def format_metric_value(value: Any, metric_type: str = "general") -> str:
    """Convenience function for formatting metric values"""
    
    formatters = {
        "percentage": lambda v: DataFormatter.format_percentage(v),
        "score": lambda v: DataFormatter.format_score(v),
        "confidence": lambda v: DataFormatter.format_confidence(v),
        "number": lambda v: DataFormatter.format_number(v),
        "duration": lambda v: DataFormatter.format_duration(v),
        "size": lambda v: DataFormatter.format_file_size(int(v)) if isinstance(v, (int, float)) else str(v)
    }
    
    formatter = formatters.get(metric_type, str)
    return formatter(value)

def safe_format(value: Any, fallback: str = "N/A") -> str:
    """Safely format any value with fallback"""
    
    try:
        if pd.isna(value) or value is None:
            return fallback
        return str(value)
    except Exception:
        return fallback
