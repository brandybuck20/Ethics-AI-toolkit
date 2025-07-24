import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def main():
    """Render the home dashboard page"""

    # Initialize session state for audit_stats if not already present
    if 'audit_stats' not in st.session_state:
        st.session_state.audit_stats = {
            'total_audits': 0,
            'recent_audits': 0,
            'issues_found': 0,
            'avg_ethics_score': 0.0,
            'last_audit_date': None,
            'audit_history': []
        }
    
    st.title("🛡️ AI Ethics Toolkit Dashboard")
    st.markdown("**Comprehensive overview of your AI ethics auditing activities**")
    
    # Quick action buttons
    render_quick_actions()
    
    st.markdown("---")
    
    # Key metrics overview
    render_key_metrics()
    
    # Charts and analytics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_audit_trends()
    
    with col2:
        render_recent_activity()
    
    # System health and getting started
    render_system_status()
    
    # Show getting started guide for new users
    if st.session_state.audit_stats['total_audits'] == 0:
        render_getting_started_guide()

def render_quick_actions():
    """Render quick action buttons for immediate access"""
    
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 **New Bias Audit**", use_container_width=True, type="primary"):
            st.switch_page("pages/02_Bias_Audit.py")
    
    with col2:
        if st.button("🔒 **Privacy Check**", use_container_width=True):
            st.switch_page("pages/03_Privacy_Analysis.py")
    
    with col3:
        if st.button("💡 **Explainability**", use_container_width=True):
            st.switch_page("pages/04_Explainability.py")
    
    with col4:
        if st.button("🔎 **Hallucination Test**", use_container_width=True):
            st.switch_page("pages/05_Hallucination.py")

def render_key_metrics():
    """Render key performance indicators"""
    
    st.subheader("📊 Key Metrics")
    
    stats = st.session_state.audit_stats
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_audits = f"+{stats.get('recent_audits', 0)}" if stats.get('recent_audits', 0) > 0 else None
        st.metric(
            "Total Audits",
            stats['total_audits'],
            delta=delta_audits
        )
    
    with col2:
        delta_issues = f"+{stats['issues_found']}" if stats['issues_found'] > 0 else "0"
        st.metric(
            "Issues Detected",
            stats['issues_found'],
            delta=delta_issues,
            delta_color="inverse"
        )
    
    with col3:
        ethics_score = stats['avg_ethics_score']
        delta_score = "+0.5" if ethics_score > 7.0 else None
        color = "normal" if ethics_score > 7.0 else "inverse"
        st.metric(
            "Avg Ethics Score",
            f"{ethics_score:.1f}/10",
            delta=delta_score,
            delta_color=color
        )
    
    with col4:
        last_audit = stats.get('last_audit_date')
        if last_audit:
            if isinstance(last_audit, str):
                last_audit = datetime.fromisoformat(last_audit.replace('Z', '+00:00'))
            days_ago = (datetime.now() - last_audit).days
            st.metric(
                "Last Audit",
                f"{days_ago} days ago",
                delta=None
            )
        else:
            st.metric("Last Audit", "Never", delta=None)

def render_audit_trends():
    """Render audit trends and analytics"""
    
    st.subheader("📈 Ethics Score Trends")
    
    # Generate trend data based on audit history
    audit_history = st.session_state.audit_stats.get('audit_history', [])
    
    if len(audit_history) > 0:
        # Convert history to DataFrame
        df_history = pd.DataFrame(audit_history)
        df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date
        
        # Group by date and calculate average scores
        daily_scores = df_history.groupby('date')['ethics_score'].mean().reset_index()
    else:
        # Sample data for demonstration
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='D')
        daily_scores = pd.DataFrame({
            'date': dates.date,
            'ethics_score': [7.5 + np.sin(i/5) * 1.5 + np.random.normal(0, 0.3) for i in range(len(dates))]
        })
    
    # Create trend chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_scores['date'],
        y=daily_scores['ethics_score'],
        mode='lines+markers',
        name='Ethics Score',
        line=dict(color='#2E5AAC', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x}<br><b>Score:</b> %{y:.1f}/10<extra></extra>'
    ))
    
    # Add target line
    fig.add_hline(y=8.0, line_dash="dash", line_color="green", 
                  annotation_text="Target Score (8.0)")
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Ethics Score",
        yaxis=dict(range=[0, 10]),
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_recent_activity():
    """Render recent audit activity feed"""
    
    st.subheader("🕒 Recent Activity")
    
    audit_history = st.session_state.audit_stats.get('audit_history', [])
    
    if len(audit_history) > 0:
        # Show last 5 activities
        recent_activities = sorted(audit_history, key=lambda x: x['timestamp'], reverse=True)[:5]
        
        for activity in recent_activities:
            timestamp = activity['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            time_ago = get_time_ago(timestamp)
            
            status_icon = "✅" if activity['ethics_score'] > 7.0 else "⚠️" if activity['ethics_score'] > 5.0 else "❌"
            status_class = "status-success" if activity['ethics_score'] > 7.0 else "status-warning" if activity['ethics_score'] > 5.0 else "status-danger"
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 12px; padding: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <strong>{activity['type'].title()} Audit</strong><br>
                        <small style="color: #64748B;">{activity.get('model_type', 'Unknown Model')}</small><br>
                        <span class="status-badge {status_class}">{status_icon} Score: {activity['ethics_score']:.1f}/10</span>
                    </div>
                    <div style="text-align: right; color: #64748B; font-size: 12px;">
                        {time_ago}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Sample activities for demonstration
        sample_activities = [
            {"type": "Bias", "model": "RandomForest Classifier", "status": "✅ Passed", "score": 8.2, "time": "2 hours ago"},
            {"type": "Privacy", "model": "GPT-4 Model", "status": "⚠️ Warnings", "score": 6.5, "time": "5 hours ago"},
            {"type": "Hallucination", "model": "Custom LLM", "status": "❌ Issues Found", "score": 4.8, "time": "1 day ago"},
            {"type": "Explainability", "model": "XGBoost Model", "status": "✅ Passed", "score": 9.1, "time": "2 days ago"},
        ]
        
        for activity in sample_activities:
            status_class = "status-success" if "✅" in activity['status'] else "status-warning" if "⚠️" in activity['status'] else "status-danger"
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 12px; padding: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <strong>{activity['type']} Audit</strong><br>
                        <small style="color: #64748B;">{activity['model']}</small><br>
                        <span class="status-badge {status_class}">{activity['status']}</span>
                    </div>
                    <div style="text-align: right; color: #64748B; font-size: 12px;">
                        {activity['time']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_system_status():
    """Render system health indicators"""
    
    st.markdown("---")
    st.subheader("🔧 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Status", "🟢 Online", help="All services operational")
    
    with col2:
        model_count = len(['sklearn', 'xgboost', 'huggingface', 'openai', 'anthropic'])
        st.metric("Model Types", f"{model_count} Supported", help="Supported ML frameworks")
    
    with col3:
        check_count = 50  # Total number of ethics checks
        st.metric("Ethics Checks", f"{check_count}+ Available", help="Comprehensive audit capabilities")
    
    with col4:
        st.metric("Version", "v1.0.0", help="Current toolkit version")

def render_getting_started_guide():
    """Render getting started guide for new users"""
    
    st.markdown("---")
    st.subheader("🚀 Getting Started Guide")
    
    st.info("""
    **Welcome to the AI Ethics Toolkit!** Follow these steps to perform your first audit:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Prepare Your Assets
        - **Model File**: Export your trained model (.pkl, .joblib)
        - **Test Dataset**: Prepare CSV with features and outcomes
        - **API Keys**: Set up OpenAI/Anthropic keys for LLM audits
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Choose Your Audit
        - **Bias Detection**: Check for demographic discrimination
        - **Privacy Analysis**: Scan for data leaks and PII exposure
        - **Explainability**: Generate model interpretations
        - **Hallucination Check**: Test LLMs for false information
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Review & Act
        - **Detailed Reports**: Get comprehensive audit results
        - **Actionable Insights**: Follow our recommendations
        - **Export Documentation**: Generate compliance reports
        - **Continuous Monitoring**: Set up ongoing checks
        """)
    
    # Quick start button
    st.markdown("### 🎯 Ready to Start?")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 **Start Your First Audit**", use_container_width=True, type="primary"):
            st.switch_page("pages/02_📊_Bias_Audit.py")

def get_time_ago(timestamp):
    """Convert timestamp to human-readable 'time ago' format"""
    now = datetime.now()
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hours ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "Just now"

if __name__ == "__main__":
    main()
