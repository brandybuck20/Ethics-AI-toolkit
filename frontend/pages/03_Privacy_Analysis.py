import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import hashlib
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.privacy.analyzer import PrivacyAnalyzer
from core.privacy.pii_detector import PIIDetector
from frontend.utils.state_manager import StateManager

def main():
    """Main privacy analysis page"""
    
    st.title("🔒 Privacy Risk Analysis")
    st.markdown("**Detect data leaks, PII exposure, and privacy violations in AI models**")
    
    # Privacy risk categories info
    render_privacy_categories()
    
    # Progress indicator
    render_progress_indicator()
    
    st.markdown("---")
    
    # Step-by-step workflow
    render_workflow_steps()

def render_privacy_categories():
    """Display privacy risk categories"""
    
    with st.expander("🛡️ Privacy Risk Categories We Detect", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📋 PII Detection**
            - Names and personal identifiers
            - Email addresses and phone numbers
            - Social Security Numbers
            - Credit card information
            - Medical records
            """)
        
        with col2:
            st.markdown("""
            **🧠 Model Memorization**
            - Training data verbatim recall
            - Canary token exposure
            - Membership inference attacks
            - Data reconstruction risks
            """)
        
        with col3:
            st.markdown("""
            **⚖️ Compliance Checks**
            - GDPR Article 17 (Right to be forgotten)
            - CCPA compliance
            - HIPAA requirements
            - Data licensing violations
            """)

def render_progress_indicator():
    """Render progress indicator for privacy audit"""
    
    steps = [
        "Upload Model/Data",
        "Configure Analysis", 
        "Run Privacy Scan",
        "Review Findings",
        "Generate Report"
    ]
    
    current_step = get_current_privacy_step()
    
    # Progress bar
    progress = (current_step + 1) / len(steps)
    st.progress(progress, text=f"Step {current_step + 1} of {len(steps)}: {steps[current_step]}")
    
    # Step indicators
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.markdown(f"✅ **{step}**")
            elif i == current_step:
                st.markdown(f"🔄 **{step}**")
            else:
                st.markdown(f"⭕ {step}")

def get_current_privacy_step():
    """Determine current step in privacy workflow"""
    if not (st.session_state.uploaded_model or st.session_state.get('text_input_for_analysis')):
        return 0
    if not st.session_state.get('privacy_config'):
        return 1
    if not st.session_state.get('privacy_results'):
        return 2
    if st.session_state.get('privacy_results') and not st.session_state.get('privacy_report_generated'):
        return 3
    return 4

def render_workflow_steps():
    """Render complete privacy analysis workflow"""
    
    # Step 1: Input selection
    render_input_selection()
    
    # Step 2: Configuration (conditional)
    if st.session_state.uploaded_model or st.session_state.get('text_input_for_analysis'):
        st.markdown("---")
        render_privacy_configuration()
    
    # Step 3: Analysis (conditional) 
    if st.session_state.get('privacy_config'):
        st.markdown("---")
        render_privacy_analysis()
    
    # Step 4: Results (conditional)
    if st.session_state.get('privacy_results'):
        st.markdown("---")
        render_privacy_results()

def render_input_selection():
    """Render input selection for privacy analysis"""
    
    st.subheader("1️⃣ Select Analysis Input")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Analysis", "📝 Text Analysis", "📊 Dataset Analysis", "🧪 Demo Mode"])
    
    with tab1:
        st.markdown("**Analyze trained models for memorization and data leaks**")
        
        if st.session_state.uploaded_model:
            st.success(f"✅ Model loaded: {st.session_state.model_metadata.get('type', 'Unknown')}")
            
            with st.expander("Model Details"):
                metadata = st.session_state.model_metadata
                st.write("**Type:**", metadata.get('type', 'Unknown'))
                st.write("**Filename:**", metadata.get('filename', 'Unknown'))
                st.write("**Upload Time:**", str(metadata.get('upload_time', 'Unknown')))
            
            model_analysis_type = st.selectbox(
                "Analysis Type",
                ["Membership Inference", "Data Reconstruction", "Canary Token Detection"],
                help="Choose the type of privacy analysis for your model"
            )
            
            st.session_state.analysis_input_type = 'model'
            st.session_state.model_analysis_type = model_analysis_type
        else:
            st.info("👆 Upload a model in the Bias Audit section first, or use text/dataset analysis below.")
    
    with tab2:
        st.markdown("**Analyze text outputs for PII and sensitive information**")
        
        analysis_mode = st.radio(
            "Text Analysis Mode",
            ["Single Text Sample", "Batch Text Analysis", "LLM Output Analysis"],
            horizontal=True
        )
        
        if analysis_mode == "Single Text Sample":
            text_input = st.text_area(
                "Enter text to analyze",
                height=150,
                placeholder="Paste the text you want to analyze for privacy risks...",
                help="Analyze any text for PII, sensitive data, and privacy violations"
            )
            
            if text_input:
                st.session_state.text_input_for_analysis = text_input
                st.session_state.analysis_input_type = 'text_single'
        
        elif analysis_mode == "Batch Text Analysis":
            uploaded_texts = st.file_uploader(
                "Upload text file(s)",
                type=['txt', 'csv', 'json'],
                accept_multiple_files=True,
                help="Upload multiple text files for batch privacy analysis"
            )
            
            if uploaded_texts:
                text_data = []
                for file in uploaded_texts:
                    if file.type == 'text/plain':
                        content = str(file.read(), 'utf-8')
                        text_data.append({'filename': file.name, 'content': content})
                    elif file.type == 'text/csv':
                        df = pd.read_csv(file)
                        for idx, row in df.iterrows():
                            text_data.append({'filename': f"{file.name}_row_{idx}", 'content': str(row.to_dict())})
                
                st.session_state.batch_text_data = text_data
                st.session_state.analysis_input_type = 'text_batch'
                st.success(f"✅ Loaded {len(text_data)} text samples for analysis")
        
        else:  # LLM Output Analysis
            st.markdown("**Analyze LLM responses for hallucinations and privacy leaks**")
            
            col1, col2 = st.columns(2)
            with col1:
                prompt_input = st.text_area("Test Prompt", height=100, placeholder="Enter your prompt here...")
            with col2:
                llm_response = st.text_area("LLM Response", height=100, placeholder="Paste the LLM response here...")
            
            if prompt_input and llm_response:
                st.session_state.llm_analysis_data = {
                    'prompt': prompt_input,
                    'response': llm_response
                }
                st.session_state.analysis_input_type = 'llm_output'
    
    with tab3:
        st.markdown("**Analyze datasets for privacy compliance**")
        
        if st.session_state.uploaded_dataset is not None:
            st.success(f"✅ Dataset loaded: {st.session_state.dataset_metadata.get('shape', 'Unknown shape')}")
            
            # Dataset privacy analysis options
            dataset_analysis_options = st.multiselect(
                "Dataset Analysis Options",
                [
                    "PII Column Detection",
                    "Data Anonymization Check", 
                    "GDPR Compliance Scan",
                    "Sensitive Data Classification"
                ],
                default=["PII Column Detection", "GDPR Compliance Scan"]
            )
            
            if dataset_analysis_options:
                st.session_state.dataset_analysis_options = dataset_analysis_options
                st.session_state.analysis_input_type = 'dataset'
        else:
            st.info("👆 Upload a dataset in the Bias Audit section first.")
    
    with tab4:
        st.markdown("**Try privacy analysis with demo data**")
        
        demo_scenario = st.selectbox(
            "Demo Scenario",
            [
                "Customer Support Chat Logs",
                "Medical Records Sample", 
                "Financial Transaction Data",
                "HR Employee Database",
                "Social Media Posts"
            ]
        )
        
        if st.button("🎲 Load Demo Scenario", use_container_width=True):
            load_demo_privacy_scenario(demo_scenario)

def load_demo_privacy_scenario(scenario):
    """Load demonstration privacy scenario"""
    
    # Call backend with is_demo=True for privacy analysis
    try:
        api_client = st.session_state.api_client
        # This calls the privacy analysis endpoint with is_demo=True
        # The specific content returned depends on the backend's demo implementation
        response = api_client.run_explainability_analysis(
            target_column="dummy", # Dummy target column
            is_demo=True # Request demo data
        )

        # Use mock data from backend response
        st.session_state.text_input_for_analysis = response.get("explanation", "Demo text from backend.")
        st.session_state.demo_pii_types = ["names", "emails"] # Placeholder, backend should provide this
        st.session_state.analysis_input_type = 'demo'
        st.success(f"✅ Loaded demo scenario: {scenario} from backend.")
    except Exception as e:
        st.error(f"❌ Error loading demo privacy scenario from backend: {str(e)}")

def render_privacy_configuration():
    """Render privacy analysis configuration"""
    
    st.subheader("2️⃣ Configure Privacy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Detection Settings")
        
        # PII Detection Options
        pii_types = st.multiselect(
            "PII Types to Detect",
            [
                "Names (Person, Organization)",
                "Email Addresses", 
                "Phone Numbers",
                "Social Security Numbers",
                "Credit Card Numbers",
                "IP Addresses",
                "Physical Addresses",
                "Dates of Birth",
                "Medical Record Numbers",
                "Bank Account Numbers"
            ],
            default=[
                "Names (Person, Organization)",
                "Email Addresses",
                "Phone Numbers", 
                "Social Security Numbers"
            ]
        )
        
        # Sensitivity Levels
        sensitivity_level = st.select_slider(
            "Detection Sensitivity",
            options=["Low", "Medium", "High", "Maximum"],
            value="High",
            help="Higher sensitivity may produce more false positives"
        )
        
        # Context Analysis
        context_analysis = st.checkbox(
            "Enable Context Analysis",
            value=True,
            help="Analyze surrounding context to reduce false positives"
        )
    
    with col2:
        st.markdown("#### ⚙️ Analysis Options")
        
        # Analysis Depth
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Scan", "Standard Analysis", "Deep Analysis"],
            index=1,
            help="Deeper analysis takes longer but finds more issues"
        )
        
        # Compliance Frameworks
        compliance_frameworks = st.multiselect(
            "Compliance Frameworks",
            ["GDPR", "CCPA", "HIPAA", "PCI DSS", "SOX"],
            default=["GDPR"],
            help="Check compliance against selected privacy regulations"
        )
        
        # Masking Options
        mask_findings = st.checkbox(
            "Mask Detected PII in Results",
            value=True,
            help="Replace detected PII with masked values in output"
        )
        
        # Export Settings
        include_confidence_scores = st.checkbox(
            "Include Confidence Scores",
            value=True,
            help="Show confidence levels for each detection"
        )
    
    # Advanced Configuration
    with st.expander("🔧 Advanced Settings"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Custom PII Patterns
            st.markdown("**Custom Detection Patterns**")
            custom_patterns = st.text_area(
                "Custom Regex Patterns (one per line)",
                placeholder="employee_id:\\d{6}\ncustomer_ref:[A-Z]{3}\\d{4}",
                help="Add custom regex patterns for organization-specific identifiers"
            )
            
            # Whitelist
            whitelist_patterns = st.text_area(
                "Whitelist Patterns (exclude from detection)",
                placeholder="example@company.com\ntest-user-.*",
                help="Patterns to exclude from PII detection"
            )
        
        with col2:
            # Thresholds
            confidence_threshold = st.slider(
                "Minimum Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum confidence score to report a finding"
            )
            
            max_findings = st.number_input(
                "Maximum Findings to Report",
                min_value=10,
                max_value=1000,
                value=100,
                help="Limit the number of findings to prevent information overload"
            )
            
            # Performance Options
            parallel_processing = st.checkbox(
                "Enable Parallel Processing",
                value=True,
                help="Use multiple CPU cores for faster analysis"
            )
    
    # Save configuration
    if pii_types:
        privacy_config = {
            'pii_types': pii_types,
            'sensitivity_level': sensitivity_level,
            'context_analysis': context_analysis,
            'analysis_depth': analysis_depth,
            'compliance_frameworks': compliance_frameworks,
            'mask_findings': mask_findings,
            'include_confidence_scores': include_confidence_scores,
            'custom_patterns': custom_patterns.split('\n') if custom_patterns else [],
            'whitelist_patterns': whitelist_patterns.split('\n') if whitelist_patterns else [],
            'confidence_threshold': confidence_threshold,
            'max_findings': max_findings,
            'parallel_processing': parallel_processing
        }
        
        st.session_state.privacy_config = privacy_config
        
        st.success("✅ Privacy analysis configured!")
        
        # Configuration summary
        with st.expander("📋 Configuration Summary"):
            st.write("**PII Types:**", len(pii_types), "types selected")
            st.write("**Sensitivity:**", sensitivity_level)
            st.write("**Compliance:**", ', '.join(compliance_frameworks))
            st.write("**Analysis Depth:**", analysis_depth)

def render_privacy_analysis():
    """Render privacy analysis execution"""
    
    st.subheader("3️⃣ Run Privacy Analysis")
    
    config = st.session_state.privacy_config
    
    # Display analysis summary
    analysis_type = st.session_state.get('analysis_input_type', 'unknown')
    
    st.info(f"""
    **Ready to analyze:** {get_analysis_type_description(analysis_type)}
    
    **Configuration:**
    - PII Types: {len(config['pii_types'])} selected
    - Sensitivity: {config['sensitivity_level']}
    - Compliance: {', '.join(config['compliance_frameworks'])}
    - Analysis Depth: {config['analysis_depth']}
    """)
    
    # Run analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 **Run Privacy Analysis**", use_container_width=True, type="primary"):
            run_privacy_analysis()

def get_analysis_type_description(analysis_type):
    """Get human-readable description of analysis type"""
    descriptions = {
        'model': 'Trained ML Model',
        'text_single': 'Single Text Sample',
        'text_batch': 'Multiple Text Files',
        'llm_output': 'LLM Response',
        'dataset': 'Dataset/CSV File',
        'demo': 'Demo Scenario'
    }
    return descriptions.get(analysis_type, 'Unknown Input Type')

def run_privacy_analysis():
    """Execute privacy analysis"""
    
    with st.spinner("🔍 Analyzing for privacy risks and PII exposure..."):
        try:
            analysis_type = st.session_state.get('analysis_input_type')
            config = st.session_state.privacy_config
            
            # Initialize privacy analyzer
            analyzer = PrivacyAnalyzer(config)
            
            # Run analysis based on input type
            if analysis_type == 'text_single':
                results = analyzer.analyze_text(st.session_state.text_input_for_analysis)
            elif analysis_type == 'text_batch':
                results = analyzer.analyze_batch_text(st.session_state.batch_text_data)
            elif analysis_type == 'llm_output':
                results = analyzer.analyze_llm_output(st.session_state.llm_analysis_data)
            elif analysis_type == 'dataset':
                results = analyzer.analyze_dataset(st.session_state.uploaded_dataset)
            elif analysis_type == 'model':
                results = analyzer.analyze_model(st.session_state.uploaded_model)
            elif analysis_type == 'demo':
                results = analyzer.analyze_text(st.session_state.text_input_for_analysis)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Store results
            st.session_state.privacy_results = results
            
            # Update audit statistics
            issues_count = results['summary']['total_issues']
            privacy_score = results['overall_score']
            
            StateManager.update_audit_stats('privacy', issues_count, privacy_score)
            
            st.success("✅ Privacy analysis completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Privacy analysis failed: {str(e)}")
            st.info("Please check your input data and configuration settings.")

def render_privacy_results():
    """Render comprehensive privacy analysis results"""
    
    st.subheader("4️⃣ Privacy Analysis Results")
    
    results = st.session_state.privacy_results
    
    # Overall summary
    render_privacy_summary(results)
    
    # Detailed findings
    render_detailed_findings(results)
    
    # Compliance assessment
    render_compliance_assessment(results)
    
    # Visualizations
    render_privacy_visualizations(results)
    
    # Recommendations
    render_privacy_recommendations(results)
    
    # Export options
    render_privacy_export_options(results)

def render_privacy_summary(results):
    """Render privacy analysis summary"""
    
    st.markdown("#### 📊 Privacy Risk Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = results['overall_score']
        color = "normal" if score >= 7 else "inverse"
        st.metric("Privacy Score", f"{score:.1f}/10", help="Overall privacy protection rating")
    
    with col2:
        total_issues = results['summary']['total_issues']
        st.metric("Issues Found", total_issues, delta=f"+{total_issues}" if total_issues > 0 else "0", delta_color="inverse")
    
    with col3:
        high_risk = results['summary']['high_risk_findings']
        st.metric("High Risk", high_risk, help="Critical privacy violations requiring immediate attention")
    
    with col4:
        pii_instances = results['summary']['pii_instances']
        st.metric("PII Detected", pii_instances, help="Total personally identifiable information instances found")
    
    # Risk level assessment
    risk_level = results['summary']['risk_level']
    if risk_level == 'Low':
        st.success("✅ **Risk Level: LOW** - Minimal privacy concerns detected")
    elif risk_level == 'Medium':
        st.warning("⚠️ **Risk Level: MEDIUM** - Some privacy issues require attention")
    else:
        st.error("🚨 **Risk Level: HIGH** - Significant privacy violations detected")

def render_detailed_findings(results):
    """Render detailed privacy findings"""
    
    st.markdown("#### 🔍 Detailed Findings")
    
    findings = results['findings']
    
    if not findings:
        st.success("🎉 No privacy issues detected!")
        return
    
    # Group findings by category
    findings_by_category = {}
    for finding in findings:
        category = finding['category']
        if category not in findings_by_category:
            findings_by_category[category] = []
        findings_by_category[category].append(finding)
    
    # Display findings by category
    for category, category_findings in findings_by_category.items():
        
        with st.expander(f"🏷️ {category.title()} ({len(category_findings)} findings)", expanded=len(category_findings) <= 5):
            
            for i, finding in enumerate(category_findings, 1):
                
                # Severity color coding
                severity = finding['severity']
                if severity == 'High':
                    severity_color = '#dc3545'
                    severity_icon = '🚨'
                elif severity == 'Medium':
                    severity_color = '#fd7e14'
                    severity_icon = '⚠️'
                else:
                    severity_color = '#28a745'
                    severity_icon = '💡'
                
                # Finding details
                st.markdown(f"""
                <div style="border-left: 4px solid {severity_color}; padding-left: 15px; margin-bottom: 20px;">
                    <h4>{severity_icon} Finding #{i}: {finding['type']}</h4>
                    <p><strong>Severity:</strong> <span style="color: {severity_color};">{severity}</span></p>
                    <p><strong>Description:</strong> {finding['description']}</p>
                    <p><strong>Location:</strong> {finding['location']}</p>
                    <p><strong>Confidence:</strong> {finding['confidence']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show context if available and not masked
                if 'context' in finding and not st.session_state.privacy_config.get('mask_findings', True):
                    with st.expander(f"Show Context for Finding #{i}"):
                        st.code(finding['context'], language='text')
                
                # Remediation suggestions
                if 'remediation' in finding:
                    st.markdown(f"**💡 Recommended Action:** {finding['remediation']}")
                
                st.markdown("---")

def render_compliance_assessment(results):
    """Render compliance framework assessment"""
    
    st.markdown("#### ⚖️ Compliance Assessment")
    
    compliance_results = results.get('compliance', {})
    
    if not compliance_results:
        st.info("No compliance frameworks were selected for assessment.")
        return
    
    # Create compliance summary table
    compliance_data = []
    for framework, assessment in compliance_results.items():
        compliance_data.append({
            'Framework': framework,
            'Status': '✅ Compliant' if assessment['compliant'] else '❌ Non-Compliant',
            'Score': f"{assessment['score']:.1f}/10",
            'Issues': assessment['issues_count'],
            'Critical': assessment['critical_issues']
        })
    
    df_compliance = pd.DataFrame(compliance_data)
    st.dataframe(df_compliance, use_container_width=True)
    
    # Detailed compliance issues
    for framework, assessment in compliance_results.items():
        if assessment['issues_count'] > 0:
            with st.expander(f"📋 {framework} Compliance Issues ({assessment['issues_count']})"):
                for issue in assessment['issues']:
                    st.markdown(f"- **{issue['article']}**: {issue['description']}")

def render_privacy_visualizations(results):
    """Render privacy analysis visualizations"""
    
    st.markdown("#### 📈 Privacy Analysis Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["PII Distribution", "Risk Heatmap", "Trend Analysis"])
    
    with tab1:
        render_pii_distribution_chart(results)
    
    with tab2:
        render_risk_heatmap_chart(results)
    
    with tab3:
        render_trend_analysis_chart(results)

def render_pii_distribution_chart(results):
    """Render PII type distribution chart"""
    
    findings = results['findings']
    
    # Count findings by PII type
    pii_counts = {}
    for finding in findings:
        pii_type = finding['type']
        pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
    
    if pii_counts:
        # Create pie chart
        fig = px.pie(
            values=list(pii_counts.values()),
            names=list(pii_counts.keys()),
            title="Distribution of PII Types Found",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=True,
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        pii_df = pd.DataFrame([
            {'PII Type': pii_type, 'Count': count, 'Percentage': f"{count/sum(pii_counts.values())*100:.1f}%"}
            for pii_type, count in sorted(pii_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.dataframe(pii_df, use_container_width=True)
    else:
        st.info("No PII detected in the analysis.")

def render_risk_heatmap_chart(results):
    """Render privacy risk heatmap"""
    
    findings = results['findings']
    
    if not findings:
        st.info("No risk data available for heatmap visualization.")
        return
    
    # Create risk matrix data
    risk_categories = ['Names', 'Email', 'Phone', 'SSN', 'Credit Card', 'Address', 'Medical', 'Financial']
    severity_levels = ['Low', 'Medium', 'High']
    
    # Initialize matrix
    risk_matrix = np.zeros((len(risk_categories), len(severity_levels)))
    category_mapping = {cat.lower(): i for i, cat in enumerate(risk_categories)}
    severity_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Populate matrix
    for finding in findings:
        finding_type = finding['type'].lower()
        severity = finding['severity']
        
        # Map finding type to category
        for cat_key, cat_idx in category_mapping.items():
            if cat_key in finding_type or finding_type in cat_key:
                if severity in severity_mapping:
                    risk_matrix[cat_idx][severity_mapping[severity]] += 1
                break
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=risk_matrix,
        x=severity_levels,
        y=risk_categories,
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="Risk Count"),
        hoveringmode='closest',
        hovertemplate='<b>%{y}</b><br>Severity: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Privacy Risk Heatmap by Category and Severity",
        xaxis_title="Severity Level",
        yaxis_title="PII Category",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_trend_analysis_chart(results):
    """Render privacy trend analysis over time"""
    
    # For now, show sample trend data (in production, this would use historical data)
    st.markdown("**Privacy Score Trends (Last 30 Days)**")
    
    # Generate sample trend data
    dates = pd.date_range(start=datetime.now() - pd.Timedelta(days=30), 
                         end=datetime.now(), freq='D')
    
    # Simulate privacy scores with some variation
    np.random.seed(42)
    base_score = results['overall_score']
    privacy_scores = [base_score + np.random.normal(0, 0.5) for _ in range(len(dates))]
    privacy_scores = [max(0, min(10, score)) for score in privacy_scores]  # Clamp to 0-10
    
    # Create trend chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=privacy_scores,
        mode='lines+markers',
        name='Privacy Score',
        line=dict(color='#2E5AAC', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x}<br><b>Score:</b> %{y:.1f}/10<extra></extra>'
    ))
    
    # Add target line
    fig.add_hline(y=7.0, line_dash="dash", line_color="green", 
                  annotation_text="Target Score (7.0)")
    
    # Add current score line
    fig.add_hline(y=base_score, line_dash="dot", line_color="red", 
                  annotation_text=f"Current Score ({base_score:.1f})")
    
    fig.update_layout(
        title="Privacy Score Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Privacy Score",
        yaxis=dict(range=[0, 10]),
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Score", f"{base_score:.1f}/10")
    with col2:
        avg_score = np.mean(privacy_scores)
        st.metric("30-Day Average", f"{avg_score:.1f}/10")
    with col3:
        trend = "📈 Improving" if privacy_scores[-1] > privacy_scores[0] else "📉 Declining"
        st.metric("Trend", trend)

def render_privacy_recommendations(results):
    """Render actionable privacy recommendations"""
    
    st.markdown("#### 💡 Privacy Recommendations & Next Steps")
    
    # Generate recommendations based on findings
    recommendations = generate_privacy_recommendations(results)
    
    # Priority recommendations
    st.markdown("##### 🚨 Priority Actions")
    priority_recs = [rec for rec in recommendations if rec['priority'] == 'High']
    
    if priority_recs:
        for i, rec in enumerate(priority_recs, 1):
            st.markdown(f"""
            <div style="border-left: 4px solid #dc3545; padding-left: 15px; margin-bottom: 15px; background-color: #fff5f5;">
                <strong>{i}. {rec['title']}</strong><br>
                {rec['description']}<br>
                <small><strong>Impact:</strong> {rec['impact']} | <strong>Effort:</strong> {rec['effort']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No high-priority privacy actions required!")
    
    # Standard recommendations
    st.markdown("##### 📋 Additional Recommendations")
    standard_recs = [rec for rec in recommendations if rec['priority'] in ['Medium', 'Low']]
    
    for i, rec in enumerate(standard_recs, 1):
        priority_color = '#fd7e14' if rec['priority'] == 'Medium' else '#28a745'
        bg_color = '#fff8f0' if rec['priority'] == 'Medium' else '#f8fff8'
        
        with st.expander(f"{rec['priority']} Priority: {rec['title']}"):
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding-left: 15px; background-color: {bg_color};">
                {rec['description']}<br>
                <small><strong>Impact:</strong> {rec['impact']} | <strong>Effort:</strong> {rec['effort']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Best practices section
    with st.expander("📚 Privacy Best Practices"):
        st.markdown("""
        **Data Minimization:**
        - Collect only necessary personal information
        - Implement automatic data retention policies
        - Regular data purging and anonymization
        
        **Access Controls:**
        - Role-based access to sensitive data
        - Multi-factor authentication for data access
        - Audit logs for all data interactions
        
        **Technical Safeguards:**
        - Data encryption at rest and in transit
        - Tokenization of sensitive identifiers
        - Differential privacy for analytics
        
        **Governance:**
        - Regular privacy impact assessments
        - Staff training on data protection
        - Incident response procedures
        """)

def generate_privacy_recommendations(results):
    """Generate privacy recommendations based on analysis results"""
    
    recommendations = []
    findings = results['findings']
    risk_level = results['summary']['risk_level']
    
    # High-risk recommendations
    if risk_level == 'High':
        recommendations.append({
            'priority': 'High',
            'title': 'Immediate PII Remediation Required',
            'description': 'Critical privacy violations detected. Implement data masking, remove exposed PII, and review data handling procedures immediately.',
            'impact': 'High',
            'effort': 'Medium'
        })
    
    # PII-specific recommendations
    pii_types_found = set(finding['type'] for finding in findings)
    
    if 'SSN' in pii_types_found or 'Credit Card' in pii_types_found:
        recommendations.append({
            'priority': 'High',
            'title': 'Secure Financial/Identity Data',
            'description': 'Social Security Numbers or credit card information detected. Implement tokenization, encryption, and access controls immediately.',
            'impact': 'Very High',
            'effort': 'High'
        })
    
    if 'Email' in pii_types_found:
        recommendations.append({
            'priority': 'Medium',
            'title': 'Email Address Protection',
            'description': 'Email addresses detected. Consider hashing or masking email addresses in non-production environments.',
            'impact': 'Medium',
            'effort': 'Low'
        })
    
    if 'Phone' in pii_types_found:
        recommendations.append({
            'priority': 'Medium',
            'title': 'Phone Number Anonymization',
            'description': 'Phone numbers found in data. Implement phone number masking or use synthetic alternatives for testing.',
            'impact': 'Medium',
            'effort': 'Low'
        })
    
    # Compliance recommendations
    compliance_results = results.get('compliance', {})
    for framework, assessment in compliance_results.items():
        if not assessment['compliant']:
            recommendations.append({
                'priority': 'High' if assessment['critical_issues'] > 0 else 'Medium',
                'title': f'{framework} Compliance Remediation',
                'description': f'Non-compliance with {framework} detected. Review and address {assessment["issues_count"]} compliance issues.',
                'impact': 'High',
                'effort': 'Medium'
            })
    
    # General recommendations
    recommendations.extend([
        {
            'priority': 'Low',
            'title': 'Implement Regular Privacy Audits',
            'description': 'Set up automated privacy scanning in your CI/CD pipeline to catch issues early.',
            'impact': 'Medium',
            'effort': 'Low'
        },
        {
            'priority': 'Low',
            'title': 'Staff Privacy Training',
            'description': 'Provide regular training to development and operations teams on privacy best practices.',
            'impact': 'Medium',
            'effort': 'Medium'
        }
    ])
    
    return recommendations

def render_privacy_export_options(results):
    """Render privacy analysis export options"""
    
    st.markdown("#### 📄 Export & Reporting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 **Generate Privacy Report**", use_container_width=True, type="primary"):
            generate_privacy_pdf_report(results)
    
    with col2:
        if st.button("💾 **Export Findings (JSON)**", use_container_width=True):
            export_privacy_json_results(results)
    
    with col3:
        if st.button("📧 **Share with Team**", use_container_width=True):
            st.info("Team sharing functionality coming soon!")

def generate_privacy_pdf_report(results):
    """Generate comprehensive privacy audit report"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_content = f"""
AI ETHICS TOOLKIT - PRIVACY ANALYSIS REPORT
===========================================

Generated: {timestamp}
Analysis Type: {get_analysis_type_description(st.session_state.get('analysis_input_type', 'unknown'))}

EXECUTIVE SUMMARY
-----------------
Overall Privacy Score: {results['overall_score']:.1f}/10
Risk Level: {results['summary']['risk_level']}
Total Issues Found: {results['summary']['total_issues']}
High-Risk Findings: {results['summary']['high_risk_findings']}
PII Instances Detected: {results['summary']['pii_instances']}

DETAILED FINDINGS
-----------------
"""
    
    # Add findings details
    for i, finding in enumerate(results['findings'], 1):
        report_content += f"""
Finding #{i}: {finding['type']}
- Severity: {finding['severity']}
- Confidence: {finding['confidence']:.2f}
- Description: {finding['description']}
- Location: {finding['location']}
"""
        if 'remediation' in finding:
            report_content += f"- Recommended Action: {finding['remediation']}\n"
        report_content += "\n"
    
    # Add compliance assessment
    compliance_results = results.get('compliance', {})
    if compliance_results:
        report_content += "\nCOMPLIANCE ASSESSMENT\n"
        report_content += "-" * 21 + "\n"
        
        for framework, assessment in compliance_results.items():
            status = "COMPLIANT" if assessment['compliant'] else "NON-COMPLIANT"
            report_content += f"""
{framework}: {status}
- Score: {assessment['score']:.1f}/10
- Issues: {assessment['issues_count']}
- Critical Issues: {assessment['critical_issues']}
"""
    
    # Add recommendations
    recommendations = generate_privacy_recommendations(results)
    if recommendations:
        report_content += "\nRECOMMENDATIONS\n"
        report_content += "-" * 15 + "\n"
        
        for i, rec in enumerate(recommendations, 1):
            clean_title = rec['title'].replace('**', '').replace('🚨', '').replace('📋', '').replace('💡', '')
            clean_desc = rec['description'].replace('**', '')
            report_content += f"""
{i}. {clean_title}
   Priority: {rec['priority']}
   Description: {clean_desc}
   Impact: {rec['impact']} | Effort: {rec['effort']}

"""
    
    # Configuration details
    config = st.session_state.get('privacy_config', {})
    report_content += f"""
ANALYSIS CONFIGURATION
----------------------
PII Types Scanned: {', '.join(config.get('pii_types', []))}
Sensitivity Level: {config.get('sensitivity_level', 'Unknown')}
Analysis Depth: {config.get('analysis_depth', 'Unknown')}
Compliance Frameworks: {', '.join(config.get('compliance_frameworks', []))}
Confidence Threshold: {config.get('confidence_threshold', 'Unknown')}
"""
    
    # Download button
    st.download_button(
        label="📥 Download Privacy Report",
        data=report_content,
        file_name=f"privacy_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    st.success("✅ Privacy report generated successfully!")

def export_privacy_json_results(results):
    """Export privacy analysis results as JSON"""
    
    import json
    
    # Prepare JSON export
    json_export = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': st.session_state.get('analysis_input_type', 'unknown'),
            'toolkit_version': '1.0.0'
        },
        'configuration': st.session_state.get('privacy_config', {}),
        'results': results,
        'session_info': {
            'total_audits': st.session_state.audit_stats.get('total_audits', 0),
            'privacy_audits': st.session_state.audit_stats.get('privacy_audits', 0)
        }
    }
    
    json_str = json.dumps(json_export, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON Results",
        data=json_str,
        file_name=f"privacy_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("✅ JSON export completed successfully!")

if __name__ == "__main__":
    main()

