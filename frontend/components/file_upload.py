import streamlit as st
import pandas as pd
import joblib
import pickle
from typing import Optional, Dict, Any, List
import os
import hashlib

def render_model_uploader(
    key: str = "model_uploader",
    help_text: str = "Upload your trained ML model",
    accepted_formats: List[str] = ['pkl', 'joblib']
) -> Optional[Any]:
    """Render model file uploader with validation"""
    
    st.markdown("#### 🤖 Model Upload")
    
    # File size info
    st.markdown("""
    <div style="background: #F0F9FF; border-left: 4px solid #0EA5E9; padding: 12px; margin: 8px 0; border-radius: 4px;">
        <strong>📋 Supported Formats:</strong> .pkl, .joblib<br>
        <strong>📏 Max Size:</strong> 200MB<br>
        <strong>🔧 Compatible:</strong> scikit-learn, XGBoost, LightGBM
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose model file",
        type=accepted_formats,
        help=help_text,
        key=key
    )
    
    if uploaded_file is not None:
        return process_model_upload(uploaded_file)
    
    return None

def process_model_upload(uploaded_file) -> Optional[Dict[str, Any]]:
    """Process uploaded model file"""
    
    try:
        # Show upload progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Validate file
        status_text.text("🔍 Validating file...")
        progress_bar.progress(25)
        
        file_info = validate_uploaded_file(uploaded_file)
        if not file_info['is_valid']:
            st.error(f"❌ {file_info['error']}")
            return None
        
        # Step 2: Load model
        status_text.text("📥 Loading model...")  
        progress_bar.progress(50)
        
        model = load_model_file(uploaded_file)
        if model is None:
            st.error("❌ Failed to load model")
            return None
        
        # Step 3: Extract metadata
        status_text.text("📋 Extracting metadata...")
        progress_bar.progress(75)
        
        metadata = extract_model_metadata(model, uploaded_file)
        
        # Step 4: Complete
        status_text.text("✅ Model loaded successfully!")
        progress_bar.progress(100)
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display model info
        display_model_info(metadata)
        
        return {
            'model': model,
            'metadata': metadata,
            'file_info': file_info
        }
        
    except Exception as e:
        st.error(f"❌ Error processing model: {str(e)}")
        return None

def render_dataset_uploader(
    key: str = "dataset_uploader",
    help_text: str = "Upload your dataset for analysis",
    accepted_formats: List[str] = ['csv', 'json', 'parquet']
) -> Optional[pd.DataFrame]:
    """Render dataset uploader with validation"""
    
    st.markdown("#### 📊 Dataset Upload")
    
    # Format info
    st.markdown("""
    <div style="background: #F0FDF4; border-left: 4px solid #10B981; padding: 12px; margin: 8px 0; border-radius: 4px;">
        <strong>📋 Supported Formats:</strong> .csv, .json, .parquet<br>
        <strong>📏 Max Size:</strong> 100MB<br>
        <strong>📊 Max Rows:</strong> 100,000 rows
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose dataset file",
        type=accepted_formats,
        help=help_text,
        key=key
    )
    
    if uploaded_file is not None:
        return process_dataset_upload(uploaded_file)
    
    return None

def process_dataset_upload(uploaded_file) -> Optional[pd.DataFrame]:
    """Process uploaded dataset file"""
    
    try:
        # Show upload progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Validate file
        status_text.text("🔍 Validating file...")
        progress_bar.progress(20)
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Step 2: Load data
        status_text.text("📥 Loading data...")
        progress_bar.progress(40)
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error(f"❌ Unsupported file format: {file_extension}")
            return None
        
        # Step 3: Validate data
        status_text.text("🔍 Validating data...")
        progress_bar.progress(60)
        
        validation_result = validate_dataset(df)
        if not validation_result['is_valid']:
            st.error(f"❌ {validation_result['error']}")
            return None
        
        # Step 4: Generate summary
        status_text.text("📋 Generating summary...")
        progress_bar.progress(80)
        
        # Step 5: Complete
        status_text.text("✅ Dataset loaded successfully!")
        progress_bar.progress(100)
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display dataset info
        display_dataset_info(df, uploaded_file.name)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def validate_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Validate uploaded file"""
    
    # Check file size (200MB limit for models)
    max_size = 200 * 1024 * 1024  # 200MB in bytes
    
    if uploaded_file.size > max_size:
        return {
            'is_valid': False,
            'error': f"File too large: {uploaded_file.size / 1024 / 1024:.1f}MB (max: 200MB)"
        }
    
    # Check file extension
    allowed_extensions = ['pkl', 'joblib', 'csv', 'json', 'parquet']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_extensions:
        return {
            'is_valid': False,
            'error': f"Unsupported file type: .{file_extension}"
        }
    
    return {
        'is_valid': True,
        'size': uploaded_file.size,
        'extension': file_extension,
        'name': uploaded_file.name
    }

def load_model_file(uploaded_file) -> Optional[Any]:
    """Load model from uploaded file"""
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['pkl', 'joblib']:
            return joblib.load(uploaded_file)
        else:
            return pickle.load(uploaded_file)
            
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def extract_model_metadata(model, uploaded_file) -> Dict[str, Any]:
    """Extract metadata from loaded model"""
    
    metadata = {
        'filename': uploaded_file.name,
        'file_size': uploaded_file.size,
        'model_type': type(model).__name__,
        'upload_time': pd.Timestamp.now(),
        'file_hash': hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
    }
    
    # Extract model-specific information
    if hasattr(model, 'feature_names_in_'):
        metadata['feature_count'] = len(model.feature_names_in_)
        metadata['feature_names'] = list(model.feature_names_in_)
    
    if hasattr(model, 'classes_'):
        metadata['classes'] = list(model.classes_)
        metadata['n_classes'] = len(model.classes_)
    
    if hasattr(model, 'n_estimators'):
        metadata['n_estimators'] = model.n_estimators
    
    if hasattr(model, 'max_depth'):
        metadata['max_depth'] = model.max_depth
    
    return metadata

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataset"""
    
    # Check size limits
    max_rows = 100000
    if len(df) > max_rows:
        return {
            'is_valid': False,
            'error': f"Dataset too large: {len(df)} rows (max: {max_rows})"
        }
    
    # Check for empty dataset
    if len(df) == 0:
        return {
            'is_valid': False,
            'error': "Dataset is empty"
        }
    
    # Check column count
    if len(df.columns) == 0:
        return {
            'is_valid': False,
            'error': "Dataset has no columns"
        }
    
    return {'is_valid': True}

def display_model_info(metadata: Dict[str, Any]):
    """Display model information card"""
    
    st.success("✅ Model loaded successfully!")
    
    with st.expander("📋 Model Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {metadata['model_type']}")
            st.write(f"**File Size:** {metadata['file_size'] / 1024:.1f} KB")
            st.write(f"**Upload Time:** {metadata['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**File Hash:** {metadata['file_hash']}")
        
        with col2:
            if 'feature_count' in metadata:
                st.write(f"**Features:** {metadata['feature_count']}")
            if 'n_classes' in metadata:
                st.write(f"**Classes:** {metadata['n_classes']}")
            if 'n_estimators' in metadata:
                st.write(f"**Estimators:** {metadata['n_estimators']}")
            if 'max_depth' in metadata:
                st.write(f"**Max Depth:** {metadata['max_depth']}")
        
        # Show feature names if available
        if 'feature_names' in metadata and len(metadata['feature_names']) <= 20:
            st.write("**Feature Names:**")
            st.write(", ".join(metadata['feature_names']))
        elif 'feature_names' in metadata:
            st.write(f"**Sample Features:** {', '.join(metadata['feature_names'][:10])}... (+{len(metadata['feature_names'])-10} more)")

def display_dataset_info(df: pd.DataFrame, filename: str):
    """Display dataset information card"""
    
    st.success("✅ Dataset loaded successfully!")
    
    with st.expander("📊 Dataset Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Filename:** {filename}")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Data types summary
            dtype_counts = df.dtypes.value_counts()
            st.write("**Column Types:**")
            for dtype, count in dtype_counts.items():
                st.write(f"  - {dtype}: {count} columns")
        
        with col2:
            # Missing values
            missing_count = df.isnull().sum().sum()
            st.write(f"**Missing Values:** {missing_count}")
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.write(f"**Memory Usage:** {memory_mb:.2f} MB")
            
            # Numeric vs categorical columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            st.write(f"**Numeric Columns:** {len(numeric_cols)}")
            st.write(f"**Categorical Columns:** {len(categorical_cols)}")
        
        # Data preview
        st.write("**Data Preview:**")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Show missing values if any
        missing_by_column = df.isnull().sum()
        missing_columns = missing_by_column[missing_by_column > 0]
        
        if len(missing_columns) > 0:
            st.write("**Missing Values by Column:**")
            for col, count in missing_columns.items():
                percentage = (count / len(df)) * 100
                st.write(f"  - {col}: {count} ({percentage:.1f}%)")

def render_text_uploader(
    key: str = "text_uploader",
    help_text: str = "Upload text files for analysis",
    accepted_formats: List[str] = ['txt', 'md', 'json']
) -> Optional[str]:
    """Render text file uploader"""
    
    st.markdown("#### 📝 Text Upload")
    
    uploaded_file = st.file_uploader(
        "Choose text file",
        type=accepted_formats,
        help=help_text,
        key=key
    )
    
    if uploaded_file is not None:
        try:
            # Read text content
            if uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), 'utf-8')
            elif uploaded_file.type == "application/json":
                import json
                data = json.load(uploaded_file)
                content = json.dumps(data, indent=2)
            else:
                content = str(uploaded_file.read(), 'utf-8')
            
            # Display text info
            word_count = len(content.split())
            char_count = len(content)
            
            st.success(f"✅ Text loaded: {word_count} words, {char_count} characters")
            
            # Show preview
            with st.expander("📖 Text Preview", expanded=False):
                st.text_area("Content", content[:1000] + "..." if len(content) > 1000 else content, height=200, disabled=True)
            
            return content
            
        except Exception as e:
            st.error(f"❌ Error reading text file: {str(e)}")
            return None
    
    return None
