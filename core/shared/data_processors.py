import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing for ethics analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def preprocess_for_bias_analysis(self, 
                                   df: pd.DataFrame,
                                   protected_attributes: List[str],
                                   target_column: str) -> Dict[str, Any]:
        """
        Preprocess data specifically for bias analysis
        
        Args:
            df: Input dataframe
            protected_attributes: List of protected attribute columns
            target_column: Target variable column
            
        Returns:
            Dictionary with processed data and metadata
        """
        
        processed_df = df.copy()
        preprocessing_info = {
            'original_shape': df.shape,
            'protected_attributes': protected_attributes,
            'target_column': target_column,
            'transformations_applied': []
        }
        
        # Handle missing values in protected attributes
        for attr in protected_attributes:
            if attr in processed_df.columns:
                missing_count = processed_df[attr].isnull().sum()
                if missing_count > 0:
                    # Fill missing values with mode for categorical, median for numeric
                    if processed_df[attr].dtype == 'object':
                        fill_value = processed_df[attr].mode()[0] if not processed_df[attr].mode().empty else 'Unknown'
                        processed_df[attr].fillna(fill_value, inplace=True)
                        preprocessing_info['transformations_applied'].append(
                            f"Filled {missing_count} missing values in {attr} with mode: {fill_value}"
                        )
                    else:
                        fill_value = processed_df[attr].median()
                        processed_df[attr].fillna(fill_value, inplace=True)
                        preprocessing_info['transformations_applied'].append(
                            f"Filled {missing_count} missing values in {attr} with median: {fill_value}"
                        )
        
        # Encode protected attributes for analysis
        for attr in protected_attributes:
            if attr in processed_df.columns:
                if processed_df[attr].dtype == 'object':
                    le = LabelEncoder()
                    processed_df[f'{attr}_encoded'] = le.fit_transform(processed_df[attr].astype(str))
                    self.encoders[attr] = le
                    preprocessing_info['transformations_applied'].append(
                        f"Label encoded {attr}: {dict(zip(le.classes_, le.transform(le.classes_)))}"
                    )
        
        # Handle target variable
        if target_column in processed_df.columns:
            target_info = self._process_target_variable(processed_df, target_column)
            preprocessing_info.update(target_info)
        
        preprocessing_info['final_shape'] = processed_df.shape
        
        return {
            'processed_data': processed_df,
            'preprocessing_info': preprocessing_info,
            'encoders': self.encoders.copy()
        }
    
    def preprocess_for_privacy_analysis(self, 
                                       data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Preprocess data for privacy analysis
        
        Args:
            data: Input data (text or dataframe)
            
        Returns:
            Dictionary with processed data and metadata
        """
        
        if isinstance(data, str):
            return self._preprocess_text_for_privacy(data)
        elif isinstance(data, pd.DataFrame):
            return self._preprocess_dataframe_for_privacy(data)
        else:
            raise ValueError("Data must be string or DataFrame")
    
    def _preprocess_text_for_privacy(self, text: str) -> Dict[str, Any]:
        """Preprocess text for PII detection"""
        
        preprocessing_info = {
            'original_length': len(text),
            'transformations_applied': []
        }
        
        # Clean and normalize text
        processed_text = text
        
        # Remove extra whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        preprocessing_info['transformations_applied'].append("Normalized whitespace")
        
        # Extract sentences for better PII context
        sentences = self._extract_sentences(processed_text)
        
        # Extract potential PII contexts
        pii_contexts = self._extract_pii_contexts(processed_text)
        
        preprocessing_info.update({
            'processed_length': len(processed_text),
            'sentence_count': len(sentences),
            'potential_pii_contexts': len(pii_contexts)
        })
        
        return {
            'processed_text': processed_text,
            'sentences': sentences,
            'pii_contexts': pii_contexts,
            'preprocessing_info': preprocessing_info
        }
    
    def _preprocess_dataframe_for_privacy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess dataframe for privacy analysis"""
        
        preprocessing_info = {
            'original_shape': df.shape,
            'transformations_applied': []
        }
        
        processed_df = df.copy()
        
        # Identify potentially sensitive columns
        sensitive_columns = self._identify_sensitive_columns(processed_df)
        
        # Sample data for analysis (to avoid processing huge datasets)
        sample_size = min(1000, len(processed_df))
        if len(processed_df) > sample_size:
            processed_df = processed_df.sample(n=sample_size, random_state=42)
            preprocessing_info['transformations_applied'].append(f"Sampled {sample_size} rows from {len(df)} total rows")
        
        preprocessing_info.update({
            'final_shape': processed_df.shape,
            'sensitive_columns': sensitive_columns
        })
        
        return {
            'processed_data': processed_df,
            'sensitive_columns': sensitive_columns,
            'preprocessing_info': preprocessing_info
        }
    
    def _process_target_variable(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Process target variable for analysis"""
        
        target_info = {'target_processing': []}
        
        if target_column in df.columns:
            target_series = df[target_column]
            
            # Handle missing values in target
            missing_count = target_series.isnull().sum()
            if missing_count > 0:
                # For now, drop rows with missing targets
                df.dropna(subset=[target_column], inplace=True)
                target_info['target_processing'].append(f"Dropped {missing_count} rows with missing target values")
            
            # Encode target if categorical
            unique_values = target_series.nunique()
            if target_series.dtype == 'object' or unique_values <= 10:
                le = LabelEncoder()
                df[f'{target_column}_encoded'] = le.fit_transform(target_series.astype(str))
                self.encoders[f'{target_column}_target'] = le
                target_info['target_processing'].append(
                    f"Label encoded target: {dict(zip(le.classes_, le.transform(le.classes_)))}"
                )
                target_info['target_type'] = 'categorical'
                target_info['target_classes'] = list(le.classes_)
            else:
                target_info['target_type'] = 'continuous'
            
            target_info['target_unique_values'] = unique_values
        
        return target_info
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _extract_pii_contexts(self, text: str) -> List[Dict[str, Any]]:
        """Extract contexts that might contain PII"""
        
        pii_patterns = {
            'email_context': r'[^\s]*@[^\s]*',
            'phone_context': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'ssn_context': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card_context': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'address_context': r'\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)',
            'name_context': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        }
        
        contexts = []
        
        for context_type, pattern in pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                contexts.append({
                    'type': context_type,
                    'match': match.group(),
                    'context': context,
                    'position': (match.start(), match.end())
                })
        
        return contexts
    
    def _identify_sensitive_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify potentially sensitive columns in dataframe"""
        
        sensitive_patterns = [
            r'.*name.*', r'.*email.*', r'.*phone.*', r'.*address.*',
            r'.*ssn.*', r'.*social.*', r'.*credit.*', r'.*card.*',
            r'.*birth.*', r'.*age.*', r'.*gender.*', r'.*race.*',
            r'.*ethnicity.*', r'.*religion.*', r'.*medical.*', r'.*health.*'
        ]
        
        sensitive_columns = []
        
        for column in df.columns:
            column_lower = column.lower()
            
            # Check against patterns
            for pattern in sensitive_patterns:
                if re.match(pattern, column_lower):
                    sensitive_columns.append(column)
                    break
            
            # Check data characteristics
            if column not in sensitive_columns:
                # High cardinality might indicate identifiers
                if df[column].nunique() / len(df) > 0.9:
                    sensitive_columns.append(column)
        
        return sensitive_columns
    
    def normalize_features(self, 
                          df: pd.DataFrame, 
                          feature_columns: List[str],
                          method: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features"""
        
        processed_df = df.copy()
        numeric_features = [col for col in feature_columns 
                           if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_features:
            return processed_df
        
        if method == 'standard':
            scaler = StandardScaler()
            processed_df[numeric_features] = scaler.fit_transform(df[numeric_features])
            self.scalers['standard'] = scaler
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            processed_df[numeric_features] = scaler.fit_transform(df[numeric_features])
            self.scalers['minmax'] = scaler
        
        return processed_df
    
    def encode_categorical_features(self, 
                                  df: pd.DataFrame,
                                  categorical_columns: List[str],
                                  encoding_method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical features"""
        
        processed_df = df.copy()
        
        for column in categorical_columns:
            if column not in df.columns:
                continue
            
            if encoding_method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                processed_df = pd.concat([processed_df.drop(column, axis=1), dummies], axis=1)
            
            elif encoding_method == 'label':
                # Label encoding
                le = LabelEncoder()
                processed_df[column] = le.fit_transform(df[column].astype(str))
                self.encoders[column] = le
        
        return processed_df
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of all preprocessing operations"""
        
        return {
            'scalers_used': list(self.scalers.keys()),
            'encoders_used': list(self.encoders.keys()),
            'imputers_used': list(self.imputers.keys()),
            'total_transformations': len(self.scalers) + len(self.encoders) + len(self.imputers)
        }

class TextCleaner:
    """Text cleaning utilities for various analysis tasks"""
    
    @staticmethod
    def clean_for_analysis(text: str, preserve_case: bool = False) -> str:
        """Clean text for general analysis"""
        
        if not preserve_case:
            text = text.lower()
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        
        # Improved sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    @staticmethod
    def extract_paragraphs(text: str) -> List[str]:
        """Extract paragraphs from text"""
        
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text"""
        
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        return text
    
    @staticmethod
    def mask_potential_pii(text: str) -> str:
        """Mask potential PII in text for safe processing"""
        
        # Email masking
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone number masking
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        
        # SSN masking
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Credit card masking
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', text)
        
        return text

class DataValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataframe validation"""
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Basic checks
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("DataFrame is empty")
            return validation_results
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Check missing values
        missing_by_column = df.isnull().sum()
        high_missing_columns = missing_by_column[missing_by_column > len(df) * 0.5]
        
        if not high_missing_columns.empty:
            validation_results['warnings'].append(
                f"Columns with >50% missing values: {list(high_missing_columns.index)}"
            )
        
        # Check data types
        object_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        validation_results['summary'] = {
            'shape': df.shape,
            'duplicate_rows': duplicate_count,
            'missing_values_total': df.isnull().sum().sum(),
            'object_columns': len(object_columns),
            'numeric_columns': len(numeric_columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return validation_results
    
    @staticmethod
    def validate_protected_attributes(df: pd.DataFrame, 
                                    protected_attrs: List[str]) -> Dict[str, Any]:
        """Validate protected attributes for bias analysis"""
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'attribute_info': {}
        }
        
        for attr in protected_attrs:
            if attr not in df.columns:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Protected attribute '{attr}' not found in data")
                continue
            
            attr_series = df[attr]
            unique_values = attr_series.nunique()
            missing_count = attr_series.isnull().sum()
            
            validation_results['attribute_info'][attr] = {
                'unique_values': unique_values,
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(df)) * 100,
                'data_type': str(attr_series.dtype)
            }
            
            # Validation checks
            if unique_values < 2:
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f"Protected attribute '{attr}' has fewer than 2 unique values"
                )
            
            if missing_count > len(df) * 0.1:
                validation_results['issues'].append(
                    f"Protected attribute '{attr}' has >10% missing values"
                )
        
        return validation_results
