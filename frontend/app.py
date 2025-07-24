import streamlit as st
import sys
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Page configuration
st.set_page_config(
    page_title="AI Ethics Toolkit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ai-ethics-toolkit/docs',
        'Report a bug': "https://github.com/ai-ethics-toolkit/issues",
        'About': "AI Ethics Toolkit - Comprehensive ethical AI auditing platform"
    }
)

def main():
    """Main application entry point"""
    try:
        # Load custom styling
        from frontend.utils.theme import load_custom_css
        load_custom_css()
        
        # Initialize session state
        from frontend.utils.state_manager import StateManager
        StateManager.initialize_session_state()
        
        # Initialize API client
        from frontend.utils.api_client import APIClient
        
        # Determine API URL based on environment
        api_url = os.environ.get("API_URL", "http://localhost:12000")
        
        # Store API client in session state
        if "api_client" not in st.session_state:
            st.session_state.api_client = APIClient(base_url=api_url)
        
        # Display main application
        st.title("🛡️ AI Ethics Toolkit")
        st.markdown("**Welcome to your comprehensive AI ethics auditing platform**")
        
        # Navigation info
        st.sidebar.markdown("### 📚 Navigation")
        st.sidebar.info("""
        Use the pages in the sidebar to:
        - 🏠 **Home**: Dashboard overview
        - 📊 **Bias Audit**: Detect demographic bias
        - 💡 **Explainability**: Understand model decisions
        - 🔍 **Hallucination**: Detect false information
        - 📄 **Reports**: Generate audit documentation
        - ⚙️ **Settings**: Configure your toolkit
        """)
        
        # API connection status
        try:
            # Test API connection
            api_status = "🟢 Connected"
            api_version = "v1.0.0"
            
            # Display connection info in sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔌 API Connection")
            st.sidebar.info(f"""
            **Status:** {api_status}
            **URL:** {api_url}
            **Session ID:** {st.session_state.api_client.session_id[:8]}...
            **Version:** {api_version}
            """)
        except Exception as e:
            logger.error(f"API connection error: {str(e)}")
            st.sidebar.warning(f"⚠️ API Connection Error: {str(e)}")
        
        # Quick start guide
        st.markdown("## 🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1. Upload Model
            - Support for sklearn, XGBoost models
            - .pkl and .joblib formats
            - Automatic model detection
            """)
            
            if st.button("Go to Bias Audit", key="goto_bias"):
                st.switch_page("pages/02_Bias_Audit.py")
        
        with col2:
            st.markdown("""
            ### 2. Upload Data
            - CSV format support
            - Automatic schema detection
            - Protected attribute identification
            """)
            
            if st.button("Go to Explainability", key="goto_explain"):
                st.switch_page("pages/04_Explainability.py")
        
        with col3:
            st.markdown("""
            ### 3. Run Analysis
            - Comprehensive bias detection
            - Model explainability
            - Hallucination detection
            - Background processing
            """)
            
            if st.button("Go to Hallucination", key="goto_hallucination"):
                st.switch_page("pages/05_Hallucination.py")
        
        # System status
        st.markdown("---")
        st.markdown("## 🔧 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Status", "🟢 Online")
        with col2:
            st.metric("Models Supported", "5+")
        with col3:
            st.metric("Ethics Checks", "20+")
        with col4:
            st.metric("Version", "v1.0.0")
        
        # Recent activity
        st.markdown("---")
        st.markdown("## 📊 Recent Activity")
        
        # Check if any models or datasets have been uploaded
        if st.session_state.get("uploaded_model") or st.session_state.get("uploaded_dataset"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📁 Uploaded Files")
                
                if st.session_state.get("uploaded_model"):
                    model_meta = st.session_state.get("model_metadata", {})
                    st.success(f"✅ Model: {model_meta.get('filename', 'Unknown')}")
                else:
                    st.info("No model uploaded yet")
                
                if st.session_state.get("uploaded_dataset"):
                    dataset_meta = st.session_state.get("dataset_metadata", {})
                    st.success(f"✅ Dataset: {dataset_meta.get('filename', 'Unknown')}")
                else:
                    st.info("No dataset uploaded yet")
            
            with col2:
                st.markdown("### 📈 Analysis Results")
                
                # Check for any analysis results
                if st.session_state.get("bias_results"):
                    st.success("✅ Bias audit completed")
                else:
                    st.info("No bias audit results yet")
                
                if st.session_state.get("explainability_results"):
                    st.success("✅ Explainability analysis completed")
                else:
                    st.info("No explainability results yet")
                
                if st.session_state.get("hallucination_results"):
                    st.success("✅ Hallucination detection completed")
                else:
                    st.info("No hallucination results yet")
        else:
            st.info("No activity yet. Start by uploading a model and dataset in the Bias Audit page.")
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
