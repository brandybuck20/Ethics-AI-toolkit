from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class BiasDetectionConfig:
    """Configuration for bias detection analysis"""
    
    # Thresholds
    bias_threshold: float = 0.1
    demographic_parity_threshold: float = 0.1
    equalized_odds_threshold: float = 0.1
    equal_opportunity_threshold: float = 0.1
    calibration_threshold: float = 0.05
    
    # Metrics to calculate
    calculate_demographic_parity: bool = True
    calculate_equalized_odds: bool = True
    calculate_equal_opportunity: bool = True
    calculate_calibration: bool = True
    calculate_individual_fairness: bool = True
    
    # Analysis options
    perform_intersectional_analysis: bool = True
    min_group_size: int = 10
    confidence_level: float = 0.95
    
    # Visualization options
    generate_visualizations: bool = True
    chart_height: int = 500
    color_palette: Dict[str, str] = None
    
    # Reporting options
    include_recommendations: bool = True
    detailed_group_analysis: bool = True
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = {
                'primary': '#2E5AAC',
                'secondary': '#64748B',
                'success': '#10B981',
                'warning': '#F59E0B',
                'error': '#EF4444',
                'info': '#3B82F6'
            }

class BiasConfigManager:
    """Manage bias detection configurations"""
    
    @staticmethod
    def get_default_config() -> BiasDetectionConfig:
        """Get default bias detection configuration"""
        return BiasDetectionConfig()
    
    @staticmethod
    def get_strict_config() -> BiasDetectionConfig:
        """Get strict bias detection configuration"""
        return BiasDetectionConfig(
            bias_threshold=0.05,
            demographic_parity_threshold=0.05,
            equalized_odds_threshold=0.05,
            equal_opportunity_threshold=0.05,
            calibration_threshold=0.02,
            min_group_size=20,
            confidence_level=0.99
        )
    
    @staticmethod
    def get_lenient_config() -> BiasDetectionConfig:
        """Get lenient bias detection configuration"""
        return BiasDetectionConfig(
            bias_threshold=0.2,
            demographic_parity_threshold=0.2,
            equalized_odds_threshold=0.2,
            equal_opportunity_threshold=0.2,
            calibration_threshold=0.1,
            min_group_size=5,
            confidence_level=0.9
        )
    
    @staticmethod
    def get_intersectional_config() -> BiasDetectionConfig:
        """Get configuration optimized for intersectional analysis"""
        return BiasDetectionConfig(
            bias_threshold=0.08,
            perform_intersectional_analysis=True,
            min_group_size=15,
            detailed_group_analysis=True,
            generate_visualizations=True
        )
    
    @staticmethod
    def create_custom_config(**kwargs) -> BiasDetectionConfig:
        """Create custom bias detection configuration"""
        return BiasDetectionConfig(**kwargs)
    
    @staticmethod
    def validate_config(config: BiasDetectionConfig) -> List[str]:
        """Validate bias detection configuration"""
        
        errors = []
        
        # Validate thresholds
        if not 0 < config.bias_threshold <= 1:
            errors.append("bias_threshold must be between 0 and 1")
        
        if not 0 < config.demographic_parity_threshold <= 1:
            errors.append("demographic_parity_threshold must be between 0 and 1")
        
        if not 0 < config.equalized_odds_threshold <= 1:
            errors.append("equalized_odds_threshold must be between 0 and 1")
        
        if not 0 < config.equal_opportunity_threshold <= 1:
            errors.append("equal_opportunity_threshold must be between 0 and 1")
        
        if not 0 < config.calibration_threshold <= 1:
            errors.append("calibration_threshold must be between 0 and 1")
        
        # Validate other parameters
        if config.min_group_size < 1:
            errors.append("min_group_size must be at least 1")
        
        if not 0 < config.confidence_level < 1:
            errors.append("confidence_level must be between 0 and 1")
        
        if config.chart_height < 100:
            errors.append("chart_height must be at least 100")
        
        return errors

# Predefined configurations for common use cases
BIAS_CONFIGS = {
    'default': BiasConfigManager.get_default_config(),
    'strict': BiasConfigManager.get_strict_config(),
    'lenient': BiasConfigManager.get_lenient_config(),
    'intersectional': BiasConfigManager.get_intersectional_config()
}
