import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.explainability.explainer import ModelExplainer
from frontend.utils.state_manager import StateManager

def main():
    """Main explainability analysis page"""
    
    st.title("💡 Model Explainability Analysis")
    st.markdown("**Generate interpretable explanations for your AI model decisions**")
    
    # Explainability methods info
    render_explainability_methods()
    
    # Progress indicator
    render_progress_indicator()
    
    st.markdown("---")
    
    # Step-by-step workflow
    render_workflow_steps()

def render_explainability_methods():
    """Display available explainability methods"""
    
    with st.expander("🔍 Explainability Methods Available", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 SHAP (Global & Local)**
            - TreeExplainer for tree-based models
            - LinearExplainer for linear models
            - DeepExplainer for neural networks
            - KernelExplainer for any model
            - Feature importance rankings
            """)
        
        with col2:
            st.markdown("""
            **🔬 LIME Analysis**
            - Local interpretable explanations
            - Tabular data explanations
            - Text explanations (coming soon)
            - Image explanations (coming soon)
            - Instance-specific insights
            """)
        
        with col3:
            st.markdown("""
            **📊 Custom Methods**
            - Permutation importance
            - Partial dependence plots
            - ICE (Individual Conditional Expectation)
            - Anchor explanations
            - Counterfactual analysis
            """)

def render_progress_indicator():
    """Render progress indicator for explainability workflow"""
    
    steps = [
        "Load Model & Data",
        "Select Method", 
        "Configure Analysis",
        "Generate Explanations",
        "Review Results"
    ]
    
    current_step = get_current_explainability_step()
    
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

def get_current_explainability_step():
    """Determine current step in explainability workflow"""
    if not (st.session_state.uploaded_model and st.session_state.uploaded_dataset):
        return 0
    if not st.session_state.get('explainability_method'):
        return 1
    if not st.session_state.get('explainability_config'):
        return 2
    if not st.session_state.get('explainability_results'):
        return 3
    return 4

def render_workflow_steps():
    """Render complete explainability workflow"""
    
    # Step 1: Verify model and data
    render_model_data_verification()
    
    # Step 2: Method selection (conditional)
    if st.session_state.uploaded_model and st.session_state.uploaded_dataset:
        st.markdown("---")
        render_method_selection()
    
    # Step 3: Configuration (conditional)
    if st.session_state.get('explainability_method'):
        st.markdown("---")
        render_explainability_configuration()
    
    # Step 4: Analysis (conditional)
    if st.session_state.get('explainability_config'):
        st.markdown("---")
        render_explainability_analysis()
    
    # Step 5: Results (conditional)
    if st.session_state.get('explainability_results'):
        st.markdown("---")
        render_explainability_results()

def render_model_data_verification():
    """Verify model and data are loaded"""
    
    st.subheader("1️⃣ Model & Data Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 Model Status")
        if st.session_state.uploaded_model:
            metadata = st.session_state.model_metadata
            st.success(f"✅ Model loaded: {metadata.get('type', 'Unknown')}")
            
            with st.expander("Model Details"):
                st.write("**Type:**", metadata.get('type', 'Unknown'))
                st.write("**Filename:**", metadata.get('filename', 'Unknown'))
                if hasattr(st.session_state.uploaded_model, 'feature_names_in_'):
                    st.write("**Features:**", len(st.session_state.uploaded_model.feature_names_in_))
        else:
            st.error("❌ No model loaded")
            st.info("👆 Please upload a model in the Bias Audit section first")
    
    with col2:
        st.markdown("#### 📊 Dataset Status")
        if st.session_state.uploaded_dataset is not None:
            metadata = st.session_state.dataset_metadata
            st.success(f"✅ Dataset loaded: {metadata.get('shape', 'Unknown shape')}")
            
            with st.expander("Dataset Details"):
                st.write("**Shape:**", metadata.get('shape', 'Unknown'))
                st.write("**Columns:**", len(metadata.get('columns', [])))
                if st.session_state.get('uploaded_dataset') is not None:
                    st.write("**Sample:**")
                    st.dataframe(st.session_state.uploaded_dataset.head(3))
                elif st.session_state.get('model_metadata', {}).get('is_demo'):
                    st.info("💡 Using demo dataset for explainability. No need to upload.")
                    st.session_state.uploaded_dataset = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
                    st.session_state.dataset_metadata = {
                        'filename': 'demo_explainability_dataset.csv',
                        'dataset_id': 'demo_explainability_dataset_id',
                        'shape': st.session_state.uploaded_dataset.shape,
                        'columns': st.session_state.uploaded_dataset.columns.tolist(),
                        'upload_time': 'N/A',
                        'is_demo': True
                    }
                else:
                    st.error("❌ No dataset loaded")
                    st.info("👆 Please upload a dataset in the Bias Audit section first")

def render_method_selection():
    """Render explainability method selection"""
    
    st.subheader("2️⃣ Select Explainability Method")
    
    # Method selection tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 SHAP", "🔬 LIME", "📊 Custom Methods", "🔄 Compare All"])
    
    with tab1:
        st.markdown("**SHAP (SHapley Additive exPlanations)**")
        st.info("📖 SHAP provides unified framework for explaining predictions using game theory concepts.")
        
        # SHAP method selection
        model_type = st.session_state.model_metadata.get('type', 'Unknown')
        
        if 'RandomForest' in model_type or 'XGB' in model_type or 'Tree' in model_type:
            recommended_shap = "TreeExplainer"
            st.success(f"🎯 **Recommended:** TreeExplainer (optimized for {model_type})")
        elif 'Linear' in model_type or 'Logistic' in model_type:
            recommended_shap = "LinearExplainer"
            st.success(f"🎯 **Recommended:** LinearExplainer (optimized for {model_type})")
        else:
            recommended_shap = "KernelExplainer"
            st.info(f"🎯 **Recommended:** KernelExplainer (model-agnostic for {model_type})")
        
        shap_method = st.selectbox(
            "SHAP Explainer Type",
            ["TreeExplainer", "LinearExplainer", "KernelExplainer", "DeepExplainer"],
            index=["TreeExplainer", "LinearExplainer", "KernelExplainer", "DeepExplainer"].index(recommended_shap)
        )
        
        shap_analysis_type = st.multiselect(
            "Analysis Types",
            ["Global Feature Importance", "Local Explanations", "Interaction Effects", "Partial Dependence"],
            default=["Global Feature Importance", "Local Explanations"]
        )
        
        if st.button("🎯 Use SHAP Method", use_container_width=True, type="primary"):
            st.session_state.explainability_method = 'shap'
            st.session_state.shap_config = {
                'explainer_type': shap_method,
                'analysis_types': shap_analysis_type
            }
    
    with tab2:
        st.markdown("**LIME (Local Interpretable Model-agnostic Explanations)**")
        st.info("📖 LIME explains individual predictions by learning an interpretable model locally around the prediction.")
        
        lime_mode = st.selectbox(
            "LIME Mode",
            ["Tabular", "Text (Coming Soon)", "Image (Coming Soon)"],
            help="Select the type of data for LIME analysis"
        )
        
        if lime_mode == "Tabular":
            num_features = st.slider(
                "Number of features to explain",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of top features to include in explanations"
            )
            
            num_samples = st.slider(
                "Number of samples for local model",
                min_value=1000,
                max_value=10000,
                value=5000,
                help="Number of samples to generate for local approximation"
            )
            
            if st.button("🔬 Use LIME Method", use_container_width=True, type="primary"):
                st.session_state.explainability_method = 'lime'
                st.session_state.lime_config = {
                    'mode': lime_mode,
                    'num_features': num_features,
                    'num_samples': num_samples
                }
        else:
            st.warning("Text and Image LIME support coming in next release!")
    
    with tab3:
        st.markdown("**Custom Explainability Methods**")
        
        custom_methods = st.multiselect(
            "Select Custom Methods",
            [
                "Permutation Importance",
                "Partial Dependence Plots", 
                "Individual Conditional Expectation (ICE)",
                "Feature Interaction Analysis",
                "Counterfactual Explanations"
            ],
            default=["Permutation Importance"]
        )
        
        if custom_methods:
            if st.button("📊 Use Custom Methods", use_container_width=True, type="primary"):
                st.session_state.explainability_method = 'custom'
                st.session_state.custom_config = {
                    'methods': custom_methods
                }
    
    with tab4:
        st.markdown("**Compare Multiple Methods**")
        st.info("Generate explanations using multiple methods for comprehensive analysis")
        
        comparison_methods = st.multiselect(
            "Methods to Compare",
            ["SHAP TreeExplainer", "LIME Tabular", "Permutation Importance"],
            default=["SHAP TreeExplainer", "LIME Tabular"]
        )
        
        if len(comparison_methods) >= 2:
            if st.button("🔄 Compare All Methods", use_container_width=True, type="primary"):
                st.session_state.explainability_method = 'comparison'
                st.session_state.comparison_config = {
                    'methods': comparison_methods
                }

def render_explainability_configuration():
    """Render explainability configuration"""
    
    st.subheader("3️⃣ Configure Analysis")
    
    method = st.session_state.explainability_method
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Analysis Settings")
        
        # Sample selection
        dataset_size = len(st.session_state.uploaded_dataset)
        sample_size = st.slider(
            "Sample Size for Analysis",
            min_value=min(50, dataset_size),
            max_value=min(1000, dataset_size),
            value=min(200, dataset_size),
            help="Number of instances to analyze (smaller = faster)"
        )
        
        # Feature selection
        available_features = st.session_state.uploaded_dataset.columns.tolist()
        target_col = st.session_state.get('target_column', available_features[-1])
        feature_cols = [col for col in available_features if col != target_col]
        
        selected_features = st.multiselect(
            "Features to Analyze",
            feature_cols,
            default=feature_cols[:10] if len(feature_cols) > 10 else feature_cols,
            help="Select features for explainability analysis"
        )
        
        # Instance selection for local explanations
        if method in ['lime', 'shap']:
            explanation_instances = st.selectbox(
                "Instances for Local Explanations",
                ["Random Sample", "High Confidence Predictions", "Low Confidence Predictions", "Specific Indices"],
                help="Choose which instances to explain in detail"
            )
            
            if explanation_instances == "Specific Indices":
                specific_indices = st.text_input(
                    "Instance Indices (comma-separated)",
                    placeholder="0, 10, 25, 50",
                    help="Enter specific row indices to explain"
                )
    
    with col2:
        st.markdown("#### ⚙️ Advanced Options")
        
        # Visualization settings
        show_feature_interactions = st.checkbox(
            "Include Feature Interactions",
            value=method == 'shap',
            help="Analyze and visualize feature interactions (SHAP only)"
        )
        
        generate_summary_plots = st.checkbox(
            "Generate Summary Plots",
            value=True,
            help="Create comprehensive visualization summaries"
        )
        
        include_confidence_intervals = st.checkbox(
            "Include Confidence Intervals",
            value=True,
            help="Show uncertainty estimates in explanations"
        )
        
        # Performance settings
        parallel_processing = st.checkbox(
            "Enable Parallel Processing",
            value=True,
            help="Use multiple CPU cores for faster computation"
        )
        
        max_computation_time = st.slider(
            "Max Computation Time (minutes)",
            min_value=1,
            max_value=30,
            value=5,
            help="Maximum time to spend on analysis"
        )
        
        # Export settings
        export_individual_explanations = st.checkbox(
            "Export Individual Explanations",
            value=True,
            help="Save detailed explanations for each instance"
        )
    
    # Validate and save configuration
    if selected_features:
        config = {
            'sample_size': sample_size,
            'selected_features': selected_features,
            'target_column': target_col,
            'explanation_instances': explanation_instances,
            'show_feature_interactions': show_feature_interactions,
            'generate_summary_plots': generate_summary_plots,
            'include_confidence_intervals': include_confidence_intervals,
            'parallel_processing': parallel_processing,
            'max_computation_time': max_computation_time,
            'export_individual_explanations': export_individual_explanations
        }
        
        # Add method-specific configuration
        if method == 'shap':
            config.update(st.session_state.get('shap_config', {}))
        elif method == 'lime':
            config.update(st.session_state.get('lime_config', {}))
        elif method == 'custom':
            config.update(st.session_state.get('custom_config', {}))
        elif method == 'comparison':
            config.update(st.session_state.get('comparison_config', {}))
        
        st.session_state.explainability_config = config
        
        st.success("✅ Configuration complete!")
        
        # Configuration summary
        with st.expander("📋 Configuration Summary"):
            st.write("**Method:**", method.upper())
            st.write("**Sample Size:**", sample_size)
            st.write("**Features:**", len(selected_features))
            st.write("**Target Column:**", target_col)
            if method == 'shap':
                st.write("**SHAP Explainer:**", config.get('explainer_type', 'Unknown'))

def render_explainability_analysis():
    """Render explainability analysis execution"""
    
    st.subheader("4️⃣ Generate Explanations")
    
    config = st.session_state.explainability_config
    method = st.session_state.explainability_method
    
    # Display analysis summary
    st.info(f"""
    **Analysis Configuration:**
    - Method: {method.upper()}
    - Sample Size: {config['sample_size']} instances
    - Features: {len(config['selected_features'])} selected
    - Target: {config['target_column']}
    - Max Time: {config['max_computation_time']} minutes
    """)
    
    # Estimated computation time
    estimated_time = estimate_computation_time(config, method)
    if estimated_time > 2:
        st.warning(f"⏱️ **Estimated computation time:** {estimated_time:.1f} minutes")
    else:
        st.info(f"⏱️ **Estimated computation time:** {estimated_time:.1f} minutes")
    
    # Run analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💡 **Generate Explanations**", use_container_width=True, type="primary"):
            run_explainability_analysis()

def estimate_computation_time(config, method):
    """Estimate computation time based on configuration"""
    
    base_time = {
        'shap': 0.1,
        'lime': 0.2,
        'custom': 0.05,
        'comparison': 0.3
    }
    
    sample_factor = config['sample_size'] / 100
    feature_factor = len(config['selected_features']) / 10
    
    estimated = base_time.get(method, 0.1) * sample_factor * feature_factor
    
    return max(0.5, min(estimated, config['max_computation_time']))

def run_explainability_analysis():
    """Execute explainability analysis"""
    
    with st.spinner("💡 Generating model explanations..."):
        try:
            config = st.session_state.explainability_config
            method = st.session_state.explainability_method
            model = st.session_state.uploaded_model
            dataset = st.session_state.uploaded_dataset
            
            is_demo_mode = st.session_state.model_metadata.get('is_demo', False) or \
                           st.session_state.dataset_metadata.get('is_demo', False)

            # Initialize explainer (or directly call API for demo)
            if is_demo_mode:
                api_client = st.session_state.api_client
                results = api_client.run_explainability_analysis(
                    target_column=config['target_column'],
                    method=method,
                    sample_size=config['sample_size'],
                    num_features=config.get('num_features', 10),
                    is_demo=True
                )
            else:
                explainer = ModelExplainer(model, dataset, config)
                # Run analysis based on method
                if method == 'shap':
                    results = explainer.generate_shap_explanations()
                elif method == 'lime':
                    results = explainer.generate_lime_explanations()
                elif method == 'custom':
                    results = explainer.generate_custom_explanations()
                elif method == 'comparison':
                    results = explainer.generate_comparison_explanations()
                else:
                    raise ValueError(f"Unknown explainability method: {method}")
            
            # Store results
            st.session_state.explainability_results = results
            
            # Update audit statistics
            insights_count = len(results.get('explanations', []))
            explainability_score = results.get('overall_interpretability_score', 8.0)
            
            StateManager.update_audit_stats('explainability', 0, explainability_score)  # No "issues" for explainability
            
            st.success("✅ Explanations generated successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Explainability analysis failed: {str(e)}")
            st.info("Please check your model and data compatibility.")

def render_explainability_results():
    """Render comprehensive explainability results"""
    
    st.subheader("5️⃣ Explainability Results")
    
    results = st.session_state.explainability_results
    
    # Overall interpretability summary
    render_interpretability_summary(results)
    
    # Method-specific results
    method = st.session_state.explainability_method
    
    if method == 'shap':
        render_shap_results(results)
    elif method == 'lime':
        render_lime_results(results)
    elif method == 'custom':
        render_custom_results(results)
    elif method == 'comparison':
        render_comparison_results(results)
    
    # Export options
    render_explainability_export_options(results)

def render_interpretability_summary(results):
    """Render overall interpretability summary"""
    
    st.markdown("#### 📊 Interpretability Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = results.get('overall_interpretability_score', 8.0)
        st.metric("Interpretability Score", f"{score:.1f}/10", help="Overall model interpretability rating")
    
    with col2:
        explanations_count = len(results.get('explanations', []))
        st.metric("Explanations Generated", explanations_count, help="Total number of instance explanations")
    
    with col3:
        top_features = len(results.get('global_importance', {}).get('features', []))
        st.metric("Key Features", top_features, help="Number of important features identified")
    
    with col4:
        method = st.session_state.explainability_method
        st.metric("Method Used", method.upper(), help="Explainability method applied")
    
    # Interpretability assessment
    if score >= 8.0:
        st.success("✅ **High Interpretability** - Model decisions are well explained")
    elif score >= 6.0:
        st.warning("⚠️ **Moderate Interpretability** - Some aspects could be clearer")
    else:
        st.error("❌ **Low Interpretability** - Model behavior is difficult to understand")

def render_shap_results(results):
    """Render SHAP-specific results"""
    
    st.markdown("#### 🎯 SHAP Analysis Results")
    
    # Global feature importance
    if 'global_importance' in results:
        st.markdown("##### Global Feature Importance")
        
        importance_data = results['global_importance']
        features = importance_data['features']
        importance_values = importance_data['importance']
        
        # Create horizontal bar chart
        fig = px.bar(
            x=importance_values,
            y=features,
            orientation='h',
            title="Global Feature Importance (SHAP Values)",
            labels={'x': 'Mean |SHAP Value|', 'y': 'Features'},
            color=importance_values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP summary plot
    if 'summary_plot_data' in results:
        st.markdown("##### Feature Impact Distribution")
        
        summary_data = results['summary_plot_data']
        
        # Create SHAP-style summary plot
        fig = px.scatter(
            summary_data,
            x='shap_value',
            y='feature',
            color='feature_value',
            size='abs_shap_value',
            title="SHAP Summary Plot - Feature Impact vs Feature Value",
            labels={
                'shap_value': 'SHAP Value (Impact on Model Output)',
                'feature': 'Features',
                'feature_value': 'Feature Value'
            }
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Local explanations
    if 'local_explanations' in results:
        render_local_explanations(results['local_explanations'], "SHAP")

def render_lime_results(results):
    """Render LIME-specific results"""
    
    st.markdown("#### 🔬 LIME Analysis Results")
    
    # Local explanations (LIME's specialty)
    if 'local_explanations' in results:
        render_local_explanations(results['local_explanations'], "LIME")

def render_custom_results(results):
    """Render custom method results"""
    
    st.markdown("#### 📊 Custom Analysis Results")
    
    methods_used = results.get('methods_used', [])
    
    for method in methods_used:
        with st.expander(f"📈 {method} Results", expanded=True):
            method_results = results.get('method_results', {}).get(method, {})
            
            if method == "Permutation Importance":
                render_permutation_importance(method_results)
            elif method == "Partial Dependence Plots":
                render_partial_dependence(method_results)
            # Add other custom methods...

def render_comparison_results(results):
    """Render method comparison results"""
    
    st.markdown("#### 🔄 Method Comparison Results")
    
    methods_compared = results.get('methods_compared', [])
    
    # Comparison table
    if 'comparison_summary' in results:
        st.markdown("##### Method Performance Comparison")
        
        comparison_df = pd.DataFrame(results['comparison_summary'])
        st.dataframe(comparison_df, use_container_width=True)
    
    # Side-by-side comparison
    if len(methods_compared) == 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"##### {methods_compared[0]}")
            method1_results = results['method_results'][methods_compared[0]]
            render_method_specific_results(method1_results, methods_compared[0])
        
        with col2:
            st.markdown(f"##### {methods_compared[1]}")
            method2_results = results['method_results'][methods_compared[1]]
            render_method_specific_results(method2_results, methods_compared[1])

def render_local_explanations(explanations, method_name):
    """Render local explanations for individual instances"""
    
    st.markdown(f"##### Local Explanations ({method_name})")
    
    # Instance selector
    instance_options = [f"Instance {i}: {exp.get('prediction', 'N/A')}" 
                       for i, exp in enumerate(explanations)]
    
    selected_instance = st.selectbox(
        "Select Instance to Explain",
        range(len(explanations)),
        format_func=lambda x: instance_options[x]
    )
    
    if selected_instance < len(explanations):
        explanation = explanations[selected_instance]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature contributions
            st.markdown("**Feature Contributions**")
            
            if 'feature_importance' in explanation:
                features = explanation['feature_importance']['features']
                values = explanation['feature_importance']['values']
                
                # Create horizontal bar chart
                colors = ['red' if v < 0 else 'green' for v in values]
                
                fig = go.Figure(go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f"Feature Contributions - Instance {selected_instance}",
                    xaxis_title="Contribution to Prediction",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Instance details
            st.markdown("**Instance Details**")
            
            if 'instance_data' in explanation:
                instance_df = pd.DataFrame([explanation['instance_data']])
                st.dataframe(instance_df.T, use_container_width=True)
            
            # Prediction info
            if 'prediction' in explanation:
                st.markdown("**Prediction Information**")
                pred_info = explanation['prediction']
                st.write(f"**Predicted Value:** {pred_info}")
                
                if 'confidence' in explanation:
                    st.write(f"**Confidence:** {explanation['confidence']:.3f}")

def render_permutation_importance(results):
    """Render permutation importance results"""
    
    if 'importance_scores' in results:
        features = results['features']
        scores = results['importance_scores']
        
        fig = px.bar(
            x=scores,
            y=features,
            orientation='h',
            title="Permutation Feature Importance",
            labels={'x': 'Importance Score', 'y': 'Features'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_explainability_export_options(results):
    """Render explainability export options"""
    
    st.markdown("#### 📄 Export Explanations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 **Generate Report**", use_container_width=True, type="primary"):
            generate_explainability_report(results)
    
    with col2:
        if st.button("💾 **Export Data**", use_container_width=True):
            export_explainability_data(results)
    
    with col3:
        if st.button("📧 **Share Insights**", use_container_width=True):
            st.info("Sharing functionality coming soon!")

def generate_explainability_report(results):
    """Generate explainability analysis report"""
    
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    method = st.session_state.explainability_method
    
    report_content = f"""
AI ETHICS TOOLKIT - EXPLAINABILITY ANALYSIS REPORT
==================================================

Generated: {timestamp}
Method: {method.upper()}

INTERPRETABILITY SUMMARY
------------------------
Overall Score: {results.get('overall_interpretability_score', 'N/A')}/10
Explanations Generated: {len(results.get('explanations', []))}
Analysis Method: {method.upper()}

GLOBAL INSIGHTS
---------------
"""
    
    # Add global importance if available
    if 'global_importance' in results:
        report_content += "Top Important Features:\n"
        importance_data = results['global_importance']
        for i, (feature, importance) in enumerate(zip(importance_data['features'][:10], 
                                                     importance_data['importance'][:10]), 1):
            report_content += f"{i}. {feature}: {importance:.4f}\n"
    
    # Add method-specific insights
    config = st.session_state.explainability_config
    report_content += f"""

ANALYSIS CONFIGURATION
----------------------
Sample Size: {config['sample_size']}
Features Analyzed: {len(config['selected_features'])}
Target Variable: {config['target_column']}
Feature Interactions: {'Yes' if config.get('show_feature_interactions') else 'No'}
"""
    
    st.download_button(
        label="📥 Download Explainability Report",
        data=report_content,
        file_name=f"explainability_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    st.success("✅ Explainability report generated!")

def export_explainability_data(results):
    """Export explainability data as JSON"""
    
    import json
    
    export_data = {
        'metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'method': st.session_state.explainability_method,
            'configuration': st.session_state.explainability_config
        },
        'results': results
    }
    
    json_str = json.dumps(export_data, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON Data",
        data=json_str,
        file_name=f"explainability_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("✅ Data exported successfully!")

if __name__ == "__main__":
    main()
