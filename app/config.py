import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models" / "trained"
    DATA_DIR = PROJECT_ROOT / "data"
    REPORTS_DIR = PROJECT_ROOT / "reports" / "generated"
    
    DEFAULT_MODEL_PATH = MODELS_DIR / "model_complete.h5"
    INPUT_SIZE = (256, 256, 3)
    
    CLASS_MAPPING = {
        0: 'Benign',
        1: 'Malignant', 
        2: 'Normal'
    }
    
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
    
    EXCEL_COLORS = {
        'VP': {'fill': "D4EDDA", 'font': "155724"},
        'VN': {'fill': "D1ECF1", 'font': "0C5460"},
        'FP': {'fill': "FFF3CD", 'font': "856404"},
        'FN': {'fill': "F8D7DA", 'font': "721C24"},
        'ERROR': {'fill': "F5F5F5", 'font': "6C757D"}
    }
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.MODELS_DIR, cls.DATA_DIR, cls.REPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)