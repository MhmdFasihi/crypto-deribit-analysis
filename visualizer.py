import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path

class CryptoVolatilityOptionsVisualizer:
    """
    Comprehensive visualization system for crypto volatility and options analysis
    """
    
    def __init__(
        self,
        price_results: dict,
        options_results: dict,
        term_structures: dict,
        volatility_cones: dict,
        output_dir: str = "results"
    ) -> None:
        """Initialize visualizer with analysis results."""
        self.price_results = price_results
        self.options_results = options_results
        self.term_structures = term_structures
        self.volatility_cones = volatility_cones
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    # ... existing methods from anomaly_option.py ... 