
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve
from app.config import Config

class MetricsCalculator:
    def __init__(self):
        self.config = Config()
    
    def extract_diagnosis_from_filename(self, filename):
        """
        Extraer diagnóstico del nombre del archivo.
        CORREGIDO: Revisa palabras completas y en orden correcto.
        """
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['normal', 'norm', 'sano', 'healthy']):
            return 'Normal'
        
        elif any(word in filename_lower for word in ['benigno', 'benign', 'bueno', 'ben']):
            return 'Benign'
        
        elif any(word in filename_lower for word in ['maligno', 'malignant', 'cancer', 'malo']):
            return 'Malignant'
        
        else:
            return 'Unknown'
    
    def calculate_classification_result(self, prediction, diagnosis):
        """
        Calcular resultado de clasificación (VP, VN, FP, FN)
        Enfoque binario: Maligno vs No-Maligno (Benign + Normal)
        """

        if diagnosis == 'Unknown':
            return 'N/A'
        
        es_maligno_real = (diagnosis == 'Malignant')
        es_maligno_pred = (prediction == 'Malignant')
        
        if es_maligno_real and es_maligno_pred:
            return 'VP' 
        elif es_maligno_real and not es_maligno_pred:
            return 'FN'  
        elif not es_maligno_real and es_maligno_pred:
            return 'FP'  
        else:  
            return 'VN'  

    def calculate_real_time_metrics(self, results_df):
        """Calcular métricas en tiempo real"""
        if 'Resultado' not in results_df.columns:
            return None
        
        counts = results_df['Resultado'].value_counts()
        vp = counts.get('VP', 0)
        vn = counts.get('VN', 0) 
        fp = counts.get('FP', 0)
        fn = counts.get('FN', 0)
        total = vp + vn + fp + fn
        
        if total == 0:
            return None
        
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0
        sensitivity = vp / (vp + fn) if (vp + fn) > 0 else 0
        specificity = vn / (vn + fp) if (vn + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (vp + vn) / total
        auc = (sensitivity + specificity) / 2
        
        return {
            'VP': vp, 'VN': vn, 'FP': fp, 'FN': fn, 'TOTAL': total,
            'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity,
            'f1_score': f1_score, 'accuracy': accuracy, 'auc': auc
        }
    
    def calculate_detailed_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calcular métricas detalladas para evaluación del modelo"""
        metrics = {}
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_benign'] = precision_per_class[0] if len(precision_per_class) > 0 else 0
        metrics['precision_malignant'] = precision_per_class[1] if len(precision_per_class) > 1 else 0
        metrics['precision_normal'] = precision_per_class[2] if len(precision_per_class) > 2 else 0
        
        metrics['recall_benign'] = recall_per_class[0] if len(recall_per_class) > 0 else 0
        metrics['recall_malignant'] = recall_per_class[1] if len(recall_per_class) > 1 else 0
        metrics['recall_normal'] = recall_per_class[2] if len(recall_per_class) > 2 else 0
        
        metrics['f1_benign'] = f1_per_class[0] if len(f1_per_class) > 0 else 0
        metrics['f1_malignant'] = f1_per_class[1] if len(f1_per_class) > 1 else 0
        metrics['f1_normal'] = f1_per_class[2] if len(f1_per_class) > 2 else 0
        
        metrics['confusion_matrix'] = cm
        
        if y_pred_proba is not None:
            try:
                metrics['auc_macro'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['auc_macro'] = 0
                metrics['auc_weighted'] = 0
        
        return metrics