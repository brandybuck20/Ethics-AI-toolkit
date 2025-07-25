import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from frontend.utils.state_manager import StateManager
from frontend.components.file_upload import render_model_uploader, render_dataset_uploader

def main():
    """Main bias audit page"""
    st.title("📊 Bias & Fairness Audit")
    st.markdown("**Detect and analyze demographic bias in your AI models**")
    
    # Progress indicator
    render_progress_indicator()
    
    st.markdown("---")
    
    # Step-by-step workflow
    render_workflow_steps()

def render_progress_indicator():
    """Render progress indicator for the audit workflow"""
    
    steps = [
        "Upload Model",
        "Upload Dataset", 
        "Configure Audit",
        "Run Analysis",
        "Review Results"
    ]
    
    current_step = get_current_step()
    
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

def get_current_step():
    """Determine current step in the workflow"""
    if not st.session_state.get('uploaded_model'):
        return 0
    if not st.session_state.get('uploaded_dataset'):
        return 1
    if not st.session_state.get('protected_attributes'):
        return 2
    if not st.session_state.get('bias_results'):
        return 3
    return 4

def render_workflow_steps():
    """Render the complete workflow"""
    
    # Step 1: Model Upload
    render_model_upload()
    
    # Step 2: Dataset Upload (conditional)
    if st.session_state.get('uploaded_model'):
        st.markdown("---")
        render_dataset_upload()
    
    # Step 3: Configuration (conditional)
    if st.session_state.get('uploaded_model') and st.session_state.get('uploaded_dataset'):
        st.markdown("---")
        render_audit_configuration()
    
    # Step 4: Analysis (conditional)
    if all(st.session_state.get(key) for key in ['uploaded_model', 'uploaded_dataset', 'protected_attributes']):
        st.markdown("---")
        render_bias_analysis()
    
    # Step 5: Results (conditional)
    if st.session_state.get('bias_results'):
        st.markdown("---")
        render_results_analysis()

def render_model_upload():
    """Render model upload section"""
    
    st.subheader("1️⃣ Upload Your Model")
    
    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🔗 API Model", "🧪 Demo Model"])
    
    with tab1:
        st.markdown("**Supported formats:** .pkl, .joblib (scikit-learn, XGBoost compatible)")
        
        uploaded_file = st.file_uploader(
            "Choose a model file",
            type=['pkl', 'joblib'],
            help="Upload a trained sklearn, XGBoost, or similar model",
            key="model_uploader"
        )
        
        if uploaded_file:
            handle_model_upload(uploaded_file)
    
    with tab2:
        st.info("🚧 API model integration (OpenAI, HuggingFace) coming in next release!")
        
        # Placeholder for API integration
        api_provider = st.selectbox("Select API Provider", ["OpenAI", "HuggingFace", "Anthropic"])
        api_key = st.text_input("API Key", type="password", help="Your API key will be encrypted")
        model_id = st.text_input("Model ID", help="e.g., gpt-4, claude-3-sonnet")
        
        if st.button("Connect API Model", disabled=True):
            st.warning("Feature coming soon!")
    
    with tab3:
        st.markdown("**Try our demo model** for testing the bias audit workflow:")
        
        if st.button("🎲 Load Demo Model", use_container_width=True):
            st.session_state.bias_audit_task_id = None # Clear previous task ID
            load_demo_model() # Directly call the function

def handle_model_upload(uploaded_file):
    """Handle the model file upload"""
    try:
        # Upload model to backend
        with st.spinner("Uploading model to server..."):
            api_client = st.session_state.api_client
            response = api_client.upload_model(uploaded_file)
            
            # Store model metadata
            st.session_state.uploaded_model = True
            st.session_state.model_metadata = {
                'filename': uploaded_file.name,
                'model_id': response.get('model_id'),
                'type': response.get('model_type'),
                'upload_time': response.get('upload_time')
            }
        
        # Success message
        st.success(f"✅ Model uploaded successfully!")
        
        # Display model information
        with st.expander("📋 Model Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Type:**", st.session_state.model_metadata.get('type', 'Unknown'))
                st.write("**Filename:**", st.session_state.model_metadata.get('filename', 'Unknown'))
                st.write("**Upload Time:**", st.session_state.model_metadata.get('upload_time', 'Unknown'))
            
            with col2:
                st.write("**Model ID:**", st.session_state.model_metadata.get('model_id', 'Unknown')[:8] + "...")
                st.write("**Status:**", "Ready for analysis")
    
    except Exception as e:
        st.error(f"❌ Error uploading model: {str(e)}")
        st.info("Please ensure your model is saved using joblib or pickle and is compatible with scikit-learn.")

def load_demo_model():
    """Load a demonstration model for testing - simplified to show static results"""
    try:
        st.session_state.bias_audit_task_id = None
        
        # Simply set the session state without actually creating a model
        with st.spinner("Loading demo model..."):
            # Store model metadata directly without API call
            st.session_state.uploaded_model = True
            st.session_state.model_metadata = {
                'filename': 'demo_model.joblib',
                'model_id': "demo_model_id", # Static ID for demo
                'type': "RandomForestClassifier (Demo)",
                'upload_time': "2025-07-25T08:00:00Z", # Static timestamp
                'is_demo': True
            }
        
        # Success message
        st.success("✅ Demo model loaded successfully!")
        
        with st.expander("📋 Demo Model Information", expanded=True):
            st.write("**Model Type:** RandomForestClassifier")
            st.write("**Features:** 10 synthetic features")
            st.write("**Classes:** Binary classification (0, 1)")
            st.write("**Training Samples:** 800")
            st.write("**Test Accuracy:** ~95%")
            
        st.info("💡 **Next:** Upload your dataset or use the demo dataset in the next step.")
    except Exception as e:
        st.error(f"❌ Error creating demo model: {str(e)}")

def render_dataset_upload():
    """Render dataset upload section"""
    
    st.subheader("2️⃣ Upload Test Dataset")
    
    tab1, tab2 = st.tabs(["📁 Upload CSV", "🧪 Demo Dataset"])
    
    with tab1:
        st.markdown("**Requirements:** CSV file with features, target variable, and protected attributes")
        
        uploaded_dataset = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your test dataset with all necessary columns",
            key="dataset_uploader"
        )
        
        if uploaded_dataset:
            handle_dataset_upload(uploaded_dataset)
    
    with tab2:
        if st.session_state.model_metadata.get('is_demo'):
            if st.button("🎲 Load Demo Dataset", use_container_width=True):
                load_demo_dataset()
        else:
            st.info("Demo dataset is only available when using the demo model.")

def handle_dataset_upload(uploaded_file):
    """Handle the dataset file upload"""
    try:
        # Upload dataset to backend
        with st.spinner("Uploading dataset to server..."):
            api_client = st.session_state.api_client
            response = api_client.upload_dataset(uploaded_file)
            
            # Store dataset metadata
            st.session_state.uploaded_dataset = True
            st.session_state.dataset_metadata = {
                'filename': uploaded_file.name,
                'dataset_id': response.get('dataset_id'),
                'shape': response.get('shape'),
                'columns': response.get('columns'),
                'upload_time': response.get('upload_time')
            }
        
        # Success message
        st.success(f"✅ Dataset uploaded successfully!")
        
        # Display dataset information
        with st.expander("📋 Dataset Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                shape = st.session_state.dataset_metadata.get('shape', (0, 0))
                st.write("**Shape:**", f"{shape[0]} rows × {shape[1]} columns")
                st.write("**Filename:**", st.session_state.dataset_metadata.get('filename', 'Unknown'))
                st.write("**Upload Time:**", st.session_state.dataset_metadata.get('upload_time', 'Unknown'))
            
            with col2:
                st.write("**Dataset ID:**", st.session_state.dataset_metadata.get('dataset_id', 'Unknown')[:8] + "...")
                st.write("**Status:**", "Ready for analysis")
            
            # Column information
            columns = st.session_state.dataset_metadata.get('columns', [])
            if columns:
                st.markdown("**Columns:**")
                st.write(", ".join(columns))
    
    except Exception as e:
        st.error(f"❌ Error uploading dataset: {str(e)}")
        st.info("Please ensure your file is a valid CSV with proper formatting.")

def load_demo_dataset():
    """Load demonstration dataset - simplified to show static results"""
    try:
        # Simply set the session state without actually creating a dataset
        with st.spinner("Loading demo dataset..."):
            # Create a static list of columns for the demo dataset
            columns = ['age', 'gender', 'race', 'income', 'credit_score', 'education'] + [f'feature_{i}' for i in range(10)] + ['approved']
            
            # Store dataset metadata directly without API call
            st.session_state.uploaded_dataset = True
            st.session_state.dataset_metadata = {
                'filename': 'demo_dataset.csv',
                'dataset_id': "demo_dataset_id", # Static ID for demo
                'shape': (1000, len(columns)),
                'columns': columns,
                'upload_time': "2025-07-25T08:00:00Z", # Static timestamp
                'is_demo': True
            }
        
        # Success message
        st.success("✅ Demo dataset loaded successfully!")
        
        with st.expander("📋 Demo Dataset Information", expanded=True):
            shape = st.session_state.dataset_metadata.get('shape', (0, 0))
            st.write("**Shape:**", f"{shape[0]} rows × {shape[1]} columns")
            st.write("**Protected Attributes:** gender, race, age")
            st.write("**Target Variable:** approved (loan approval)")
            st.write("**Note:** This dataset contains intentional bias for demonstration purposes")
            
            # Column information
            columns = st.session_state.dataset_metadata.get('columns', [])
            if columns:
                st.markdown("**Columns:**")
                st.write(", ".join(columns))
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error creating demo dataset: {str(e)}")

def render_audit_configuration():
    """Render audit configuration section"""
    
    st.subheader("3️⃣ Configure Bias Audit")
    
    columns = st.session_state.dataset_metadata.get('columns', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛡️ Protected Attributes")
        
        # Suggest protected attributes based on column names
        potential_protected = []
        protected_keywords = ['gender', 'race', 'ethnicity', 'age', 'religion', 'sexual_orientation', 'disability']
        
        for col in columns:
            if any(keyword in col.lower() for keyword in protected_keywords):
                potential_protected.append(col)
        
        if potential_protected:
            st.info(f"💡 **Suggested protected attributes:** {', '.join(potential_protected)}")
        
        protected_attrs = st.multiselect(
            "Select demographic attributes to audit for bias",
            options=columns,
            default=potential_protected,
            help="Choose columns representing sensitive attributes (gender, race, age, etc.)"
        )
        
        target_column = st.selectbox(
            "Target Variable (Ground Truth)",
            options=columns,
            index=len(columns)-1 if columns else 0,  # Default to last column
            help="Select the column with actual outcomes/labels"
        )
    
    with col2:
        st.markdown("#### 📊 Audit Configuration")
        
        # Get available metrics from API
        try:
            api_client = st.session_state.api_client
            metrics_response = api_client.get_available_metrics()
            available_metrics = metrics_response.get('available_metrics', {})
            
            # Format metrics for display
            metric_options = [metric_info['name'] for metric, metric_info in available_metrics.items()]
        except Exception:
            # Fallback options if API call fails
            metric_options = [
                "Demographic Parity",
                "Equalized Odds", 
                "Equal Opportunity",
                "Calibration"
            ]
        
        fairness_metrics = st.multiselect(
            "Fairness Metrics to Calculate",
            options=metric_options,
            default=metric_options[:2] if metric_options else [],
            help="Select which fairness metrics to evaluate"
        )
        
        bias_threshold = st.slider(
            "Bias Threshold",
            min_value=0.01,
            max_value=0.30,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Maximum acceptable difference between demographic groups"
        )
        
        st.markdown("#### 🔍 Analysis Options")
        
        run_intersectional = st.checkbox(
            "Run Intersectional Analysis",
            value=True,
            help="Analyze combinations of protected attributes (e.g., race + gender)"
        )
        
        generate_recommendations = st.checkbox(
            "Generate Mitigation Recommendations",
            value=True,
            help="Suggest actions to reduce detected bias"
        )
    
    # Store configuration in session state
    if protected_attrs and target_column:
        st.session_state.protected_attributes = protected_attrs
        st.session_state.target_column = target_column
        st.session_state.fairness_metrics = fairness_metrics
        st.session_state.bias_threshold = bias_threshold
        st.session_state.run_intersectional = run_intersectional
        st.session_state.generate_recommendations = generate_recommendations
        
        # Show run button
        st.button("▶️ Run Bias Audit", type="primary", on_click=run_bias_audit, use_container_width=True)
    else:
        st.warning("⚠️ Please select at least one protected attribute and a target column to continue.")

def run_bias_audit():
    """Run the bias audit analysis"""
    
    with st.spinner("Starting bias audit..."):
        try:
            # Get data from session state
            protected_attrs = st.session_state.protected_attributes
            target_column = st.session_state.target_column
            bias_threshold = st.session_state.bias_threshold
            fairness_metrics = [m.lower().replace(' ', '_') for m in st.session_state.fairness_metrics]
            
            # Call API to run bias audit
            api_client = st.session_state.api_client
            response = api_client.run_bias_audit(
                protected_attributes=protected_attrs,
                target_column=target_column,
                bias_threshold=bias_threshold,
                fairness_metrics=fairness_metrics
            )
            
            # Store task ID
            task_id = response.get('task_id')
            st.session_state.bias_audit_task_id = task_id
            
            # Success message
            st.success(f"✅ Bias audit started! Task ID: {task_id}")
            
        except Exception as e:
            st.error(f"❌ Error starting bias audit: {str(e)}")

def render_bias_analysis():
    """Render bias analysis section"""
    
    st.subheader("4️⃣ Bias Analysis")
    
    if not st.session_state.get('bias_results'):
        # Check if there's a running task
        if st.session_state.get('bias_audit_task_id'):
            task_id = st.session_state.bias_audit_task_id
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Poll for task status
            try:
                api_client = st.session_state.api_client
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                # Wait for task completion
                with st.spinner("Running bias audit..."):
                    result = api_client.wait_for_task_completion(
                        task_id, 
                        timeout=300,
                        poll_interval=1,
                        progress_callback=update_progress
                    )
                
                # Store results in session state
                if 'result' in result:
                    st.session_state.bias_results = result['result']
                    
                    # Update audit stats
                    from frontend.utils.state_manager import StateManager
                    StateManager.StateManager.update_audit_stats(
                        audit_type="bias",
                        issues_count=len(result['result'].get('recommendations', [])),
                        ethics_score=result['result'].get('ethics_score', 0)
                    )
                    
                    # Success message
                    st.success("✅ Bias audit completed successfully!")
                    
                else:
                    st.error("❌ Task completed but no results were returned.")
            
            except Exception as e:
                st.error(f"❌ Error checking task status: {str(e)}")
        
        else:
            # Show analysis options
            st.info("Configure the audit parameters above and click 'Run Bias Audit' to start the analysis.")
            
            # Quick analysis option
            if st.button("🔍 Run Quick Analysis", help="Run a simplified analysis with default settings"):
                with st.spinner("Running quick analysis..."):
                    try:
                        # Get data from session state
                        protected_attrs = st.session_state.protected_attributes
                        target_column = st.session_state.target_column
                        
                        # Call API to run quick bias check
                        api_client = st.session_state.api_client
                        result = api_client.quick_bias_check(
                            protected_attributes=protected_attrs,
                            target_column=target_column
                        )
                        
                        # Create simplified results structure
                        quick_results = {
                            'overall_bias_score': result.get('overall_bias_score', 0),
                            'ethics_score': result.get('ethics_score', 0),
                            'bias_summary': {
                                'overall_bias_detected': result.get('bias_detected', False),
                                'bias_severity': 'Low' if result.get('ethics_score', 0) > 7 else 'Moderate' if result.get('ethics_score', 0) > 5 else 'High',
                                'primary_bias_types': [result.get('primary_bias_type', 'None')]
                            },
                            'recommendations': result.get('recommendations', []),
                            'metadata': {
                                'is_quick_analysis': True,
                                'sample_size': result.get('sample_size', 0)
                            }
                        }
                        
                        # Store results in session state
                        st.session_state.bias_results = quick_results
                        
                        # Update audit stats
                        from frontend.utils.state_manager import StateManager
                        StateManager.StateManager.update_audit_stats(
                            audit_type="bias",
                            issues_count=len(quick_results.get('recommendations', [])),
                            ethics_score=quick_results.get('ethics_score', 0)
                        )
                        
                        # Success message
                        st.success("✅ Quick analysis completed!")
                        
                        
                    except Exception as e:
                        st.error(f"❌ Error running quick analysis: {str(e)}")
    else:
        # Show analysis progress
        st.success("✅ Bias audit completed!")
        
        # Option to re-run
        if st.button("🔄 Re-run Analysis"):
            st.session_state.bias_results = None
            st.session_state.bias_audit_task_id = None
            

def render_results_analysis():
    """Render results analysis section"""
    
    st.subheader("5️⃣ Results & Recommendations")
    
    if not st.session_state.get('bias_results'):
        st.info("Run the bias audit to see results.")
        return
    
    results = st.session_state.bias_results
    
    # Ethics score gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        ethics_score = results.get('ethics_score', 0)
        bias_score = results.get('overall_bias_score', 0)
        
        # Determine score color
        if ethics_score >= 8.0:
            score_color = "green"
        elif ethics_score >= 6.0:
            score_color = "orange"
        else:
            score_color = "red"
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ethics_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Ethics Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 10], 'tickwidth': 1},
                'bar': {'color': score_color},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 7.5], 'color': "gray"},
                    {'range': [7.5, 10], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Bias summary
        st.markdown(f"**Bias Score:** {bias_score:.3f}")
        
        bias_summary = results.get('bias_summary', {})
        st.markdown(f"**Bias Severity:** {bias_summary.get('bias_severity', 'Unknown')}")
        
        if bias_summary.get('overall_bias_detected', False):
            biased_attrs = bias_summary.get('biased_attributes', 0)
            total_attrs = bias_summary.get('total_protected_attributes', 0)
            st.warning(f"⚠️ Bias detected in {biased_attrs} of {total_attrs} protected attributes")
        else:
            st.success("✅ No significant bias detected")
    
    with col2:
        # Bias by protected attribute
        st.markdown("#### Bias by Protected Attribute")
        
        # Check if we have detailed attribute analysis
        if 'protected_attribute_analysis' in results:
            # Create bar chart of bias scores by attribute
            attr_data = []
            for attr, analysis in results['protected_attribute_analysis'].items():
                attr_data.append({
                    'Attribute': attr,
                    'Bias Score': analysis['bias_score'],
                    'Has Bias': analysis['has_bias'],
                    'Bias Type': analysis['bias_type']
                })
            
            attr_df = pd.DataFrame(attr_data)
            
            # Create bar chart
            fig = px.bar(
                attr_df,
                x='Attribute',
                y='Bias Score',
                color='Has Bias',
                color_discrete_map={True: 'red', False: 'green'},
                labels={'Bias Score': 'Bias Score', 'Attribute': 'Protected Attribute'},
                title='Bias Score by Protected Attribute',
                text='Bias Score'
            )
            
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For quick analysis, show simplified view
            st.info(f"Overall bias score: {bias_score:.3f}")
            
            if 'metadata' in results and results['metadata'].get('is_quick_analysis'):
                st.info("This is a quick analysis. Run a full audit for detailed attribute-level analysis.")
    
    # Recommendations
    st.markdown("#### 📋 Recommendations")
    
    # Show recommendations
    if 'recommendations' in results and results['recommendations']:
        for i, rec in enumerate(results['recommendations']):
            st.markdown(f"{i+1}. {rec}")
    else:
        st.info("No recommendations available.")
    
    # Export options
    st.markdown("#### Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV", use_container_width=True):
            st.info("CSV export functionality will be available in the next release.")
    
    with col2:
        if st.button("📄 Generate Report", use_container_width=True):
            st.info("Report generation will be available in the next release.")

if __name__ == "__main__":
    main()