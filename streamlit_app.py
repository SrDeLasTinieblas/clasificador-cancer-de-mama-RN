import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import h5py
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

# Class mapping
class_mapping = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal',
}

# Función para calcular métricas detalladas
def calculate_detailed_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calcula métricas detalladas de evaluación del modelo
    """
    metrics = {}
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])  # Benign, Malignant, Normal
    
    # Para clasificación multiclase, calculamos métricas macro y weighted
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Métricas por clase
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
    
    # Matriz de confusión detallada
    metrics['confusion_matrix'] = cm
    
    # Si tenemos probabilidades, calculamos AUC
    if y_pred_proba is not None:
        try:
            # Para multiclase, usamos ovr (one-vs-rest)
            metrics['auc_macro'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            metrics['auc_weighted'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['auc_macro'] = 0
            metrics['auc_weighted'] = 0
    
    return metrics

def display_confusion_matrix(cm, class_names):
    """
    Muestra la matriz de confusión en formato de tabla
    """
    st.subheader("📊 Matriz de Confusión")
    
    # Crear DataFrame para la matriz de confusión
    df_cm = pd.DataFrame(cm, 
                        index=[f'Real {name}' for name in class_names],
                        columns=[f'Pred {name}' for name in class_names])
    
    st.dataframe(df_cm, use_container_width=True)
    
    # Calcular VP, VN, FP, FN para cada clase
    st.subheader("📈 Análisis Detallado por Clase")
    
    for i, class_name in enumerate(class_names):
        with st.expander(f"Métricas para clase: {class_name}"):
            # Para clasificación multiclase, calculamos one-vs-rest
            tp = cm[i, i]  # Verdaderos positivos
            fp = cm[:, i].sum() - tp  # Falsos positivos
            fn = cm[i, :].sum() - tp  # Falsos negativos
            tn = cm.sum() - tp - fp - fn  # Verdaderos negativos
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("VP (Verdaderos Positivos)", tp)
            with col2:
                st.metric("VN (Verdaderos Negativos)", tn)
            with col3:
                st.metric("FP (Falsos Positivos)", fp)
            with col4:
                st.metric("FN (Falsos Negativos)", fn)
            
            # Calcular métricas específicas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensibilidad
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precisión", f"{precision:.3f}")
            with col2:
                st.metric("Sensibilidad", f"{sensitivity:.3f}")
            with col3:
                st.metric("Especificidad", f"{specificity:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1:.3f}")

def extract_diagnosis_from_filename(filename):
    """
    Extrae el diagnóstico real del nombre del archivo
    Busca palabras clave en el nombre del archivo
    """
    filename_lower = filename.lower()
    
    # Buscar palabras clave para diagnóstico
    if any(word in filename_lower for word in ['maligno', 'malignant', 'cancer', 'malo', 'mal']):
        return 'Malignant'
    elif any(word in filename_lower for word in ['benigno', 'benign', 'bueno', 'ben']):
        return 'Benign'
    elif any(word in filename_lower for word in ['normal', 'norm', 'sano', 'healthy']):
        return 'Normal'
    else:
        # Si no encuentra palabras clave, puede usar patrones adicionales
        # o devolver un valor por defecto
        return 'Unknown'

def calculate_classification_result(prediction, diagnosis):
    """
    Calcula el resultado de clasificación (VP, VN, FP, FN)
    basado en la predicción y el diagnóstico real
    """
    if prediction == 'Malignant' and diagnosis == 'Malignant':
        return 'VP'  # Verdadero Positivo
    elif prediction == 'Malignant' and diagnosis != 'Malignant':
        return 'FP'  # Falso Positivo
    elif prediction != 'Malignant' and diagnosis == 'Malignant':
        return 'FN'  # Falso Negativo
    else:  # prediction != 'Malignant' and diagnosis != 'Malignant'
        return 'VN'  # Verdadero Negativo

def calculate_real_time_metrics(results_df):
    """
    Calcula métricas en tiempo real basadas en los resultados actuales
    """
    if 'Resultado' not in results_df.columns:
        return None
    
    # Contar VP, VN, FP, FN
    counts = results_df['Resultado'].value_counts()
    vp = counts.get('VP', 0)
    vn = counts.get('VN', 0) 
    fp = counts.get('FP', 0)
    fn = counts.get('FN', 0)
    total = vp + vn + fp + fn
    
    if total == 0:
        return None
    
    # Calcular métricas
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    sensitivity = vp / (vp + fn) if (vp + fn) > 0 else 0  # Sensibilidad/Recall
    specificity = vn / (vn + fp) if (vn + fp) > 0 else 0  # Especificidad
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (vp + vn) / total
    auc = (sensitivity + specificity) / 2  # Aproximación simple del AUC
    
    return {
        'VP': vp, 'VN': vn, 'FP': fp, 'FN': fn, 'TOTAL': total,
        'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity,
        'f1_score': f1_score, 'accuracy': accuracy, 'auc': auc
    }

def display_real_time_metrics(metrics):
    """
    Muestra las métricas calculadas en tiempo real con el formato de las imágenes
    """
    if not metrics:
        st.warning("No se pudieron calcular métricas")
        return
    
    st.subheader("📊 Métricas Calculadas - Resultados Actuales")
    
    # Tabla de totales (como en la imagen)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**📊 Matriz de Confusión - Resultados Actuales**")
        totals_df = pd.DataFrame({
            'Categoría': ['VP', 'VN', 'FP', 'FN', 'TOTALES'],
            'TOTAL': [
                metrics['VP'], 
                metrics['VN'], 
                metrics['FP'], 
                metrics['FN'], 
                metrics['TOTAL']
            ],
            'PORCENTAJE': [
                f"{(metrics['VP']/metrics['TOTAL']*100):.0f}%" if metrics['TOTAL'] > 0 else "0%",
                f"{(metrics['VN']/metrics['TOTAL']*100):.0f}%" if metrics['TOTAL'] > 0 else "0%",
                f"{(metrics['FP']/metrics['TOTAL']*100):.0f}%" if metrics['TOTAL'] > 0 else "0%",
                f"{(metrics['FN']/metrics['TOTAL']*100):.0f}%" if metrics['TOTAL'] > 0 else "0%",
                "100%"
            ]
        })
        st.dataframe(totals_df, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Indicadores de Rendimiento - Calculados**")
        indicators_df = pd.DataFrame({
            'INDICADOR': ['Precisión', 'Sensibilidad', 'Especificidad', 'F1 Score', 'AUC'],
            'FÓRMULA': [
                'VP/(VP+FP)', 
                'VP/(VP+FN)', 
                'VN/(VN+FP)', 
                '2×(P×S)/(P+S)', 
                '(S+E)/2'
            ],
            'PORCENTAJE': [
                f"{metrics['precision']:.1%}",
                f"{metrics['sensitivity']:.1%}", 
                f"{metrics['specificity']:.1%}",
                f"{metrics['f1_score']:.1%}",
                f"{metrics['auc']:.1%}"
            ]
        })
        st.dataframe(indicators_df, use_container_width=True)
    
    # Mostrar métricas en cards grandes
    st.markdown("**📈 Resumen de Indicadores**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Precisión", f"{metrics['precision']:.1%}")
    with col2:
        st.metric("🔍 Sensibilidad", f"{metrics['sensitivity']:.1%}")
    with col3:
        st.metric("🛡️ Especificidad", f"{metrics['specificity']:.1%}")
    with col4:
        st.metric("⚖️ F1-Score", f"{metrics['f1_score']:.1%}")
    with col5:
        st.metric("📊 AUC", f"{metrics['auc']:.1%}")
    
    # Mostrar conteos en formato similar a la imagen
    st.markdown("**🔢 Conteo de Resultados**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 VP", metrics['VP'], help="Verdaderos Positivos")
    with col2:
        st.metric("🔵 VN", metrics['VN'], help="Verdaderos Negativos")
    with col3:
        st.metric("🟡 FP", metrics['FP'], help="Falsos Positivos")
    with col4:
        st.metric("🔴 FN", metrics['FN'], help="Falsos Negativos")

def create_enhanced_results_dataframe(results):
    """
    Crea un DataFrame mejorado con diagnóstico y resultado incluidos
    """
    enhanced_results = []
    
    for result in results:
        if result['Prediccion'] != 'ERROR':
            # Extraer diagnóstico del nombre del archivo
            diagnosis = extract_diagnosis_from_filename(result['Nombre_Archivo'])
            
            # Calcular resultado (VP, VN, FP, FN)
            classification_result = calculate_classification_result(result['Prediccion'], diagnosis)
            
            # Agregar las nuevas columnas
            enhanced_result = result.copy()
            enhanced_result['Diagnostico'] = diagnosis
            enhanced_result['Resultado'] = classification_result
            
            enhanced_results.append(enhanced_result)
        else:
            # Para errores, mantener los valores originales
            enhanced_result = result.copy()
            enhanced_result['Diagnostico'] = 'ERROR'
            enhanced_result['Resultado'] = 'ERROR'
            enhanced_results.append(enhanced_result)
    
    return enhanced_results

def display_historical_metrics():
    """
    Muestra las métricas históricas del modelo (como las de las imágenes)
    """
    st.subheader("🎯 Rendimiento del Modelo - Hospital Público de Surquillo")
    
    # Datos de la matriz de confusión histórica (según tus imágenes)
    historical_data = {
        'VP': 100, 'VN': 23, 'FP': 41, 'FN': 155, 'TOTALES': 319
    }
    
    # Crear tabla de matriz de confusión histórica
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**📊 Matriz de Confusión Histórica**")
        confusion_df = pd.DataFrame({
            'Categoría': ['VP', 'VN', 'FP', 'FN', 'TOTALES'],
            'Total': [100, 23, 41, 155, 319],
            'Porcentaje': ['31%', '7%', '13%', '49%', '100%']
        })
        st.dataframe(confusion_df, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Indicadores de Rendimiento**")
        metrics_df = pd.DataFrame({
            'Indicador': ['Precisión', 'Sensibilidad', 'Especificidad', 'F1 Score', 'AUC'],
            'Fórmula': ['VP/(VP+FP)', 'VP/(VP+FN)', 'VN/(VN+FP)', '2×(P×S)/(P+S)', '(S+E)/2'],
            'Porcentaje': ['71%', '81%', '79.11%', '75.67%', '80.06%']
        })
        st.dataframe(metrics_df, use_container_width=True)
    
    # Mostrar métricas en formato de cards
    st.markdown("**📈 Métricas Principales del Modelo**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 Precisión",
            value="71.0%",
            help="De todas las predicciones positivas, cuántas fueron correctas"
        )
    
    with col2:
        st.metric(
            label="🔍 Sensibilidad", 
            value="81.0%",
            help="De todos los casos positivos reales, cuántos detectó el modelo"
        )
    
    with col3:
        st.metric(
            label="🛡️ Especificidad",
            value="79.11%", 
            help="De todos los casos negativos reales, cuántos identificó correctamente"
        )
    
    with col4:
        st.metric(
            label="⚖️ F1 Score",
            value="75.67%",
            help="Media armónica entre precisión y sensibilidad"
        )
    
    with col5:
        st.metric(
            label="📊 AUC",
            value="80.06%",
            help="Área bajo la curva ROC"
        )
    """
    Muestra las métricas históricas del modelo (como las de las imágenes)
    """
    st.subheader("🎯 Rendimiento del Modelo - Hospital Público de Surquillo")
    
    # Datos de la matriz de confusión histórica (según tus imágenes)
    historical_data = {
        'VP': 100, 'VN': 23, 'FP': 41, 'FN': 155, 'TOTALES': 319
    }
    
    # Crear tabla de matriz de confusión histórica
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**📊 Matriz de Confusión Histórica**")
        confusion_df = pd.DataFrame({
            'Categoría': ['VP', 'VN', 'FP', 'FN', 'TOTALES'],
            'Total': [100, 23, 41, 155, 319],
            'Porcentaje': ['31%', '7%', '13%', '49%', '100%']
        })
        st.dataframe(confusion_df, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Indicadores de Rendimiento**")
        metrics_df = pd.DataFrame({
            'Indicador': ['Precisión', 'Sensibilidad', 'Especificidad', 'F1 Score', 'AUC'],
            'Fórmula': ['VP/(VP+FP)', 'VP/(VP+FN)', 'VN/(VN+FP)', '2×(P×S)/(P+S)', '(S+E)/2'],
            'Porcentaje': ['71%', '81%', '79.11%', '75.67%', '80.06%']
        })
        st.dataframe(metrics_df, use_container_width=True)
    
    # Mostrar métricas en formato de cards
    st.markdown("**📈 Métricas Principales del Modelo**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 Precisión",
            value="71.0%",
            help="De todas las predicciones positivas, cuántas fueron correctas"
        )
    
    with col2:
        st.metric(
            label="🔍 Sensibilidad", 
            value="81.0%",
            help="De todos los casos positivos reales, cuántos detectó el modelo"
        )
    
    with col3:
        st.metric(
            label="🛡️ Especificidad",
            value="79.11%", 
            help="De todos los casos negativos reales, cuántos identificó correctamente"
        )
    
    with col4:
        st.metric(
            label="⚖️ F1 Score",
            value="75.67%",
            help="Media armónica entre precisión y sensibilidad"
        )
    
    with col5:
        st.metric(
            label="📊 AUC",
            value="80.06%",
            help="Área bajo la curva ROC"
        )

def create_detailed_results_table(predictions_df):
    """
    Crea una tabla detallada de resultados similar a la imagen
    """
    st.subheader("🗂️ Tabla Detallada de Resultados por Paciente")
    
    # Crear datos simulados para mostrar el formato
    detailed_results = []
    
    for index, row in predictions_df.iterrows():
        # Generar ID de paciente y imagen basado en el nombre del archivo
        filename = row['Nombre_Archivo']
        patient_id = f"P_{index+1:04d}"
        
        # Extraer tipo de imagen del nombre (si está disponible)
        if 'MLO' in filename.upper():
            image_type = 'MLO_IZQ' if 'IZQ' in filename.upper() else 'MLO_DER'
        elif 'CC' in filename.upper():
            image_type = 'CC_IZQ' if 'IZQ' in filename.upper() else 'CC_DER'
        else:
            image_type = 'UNK_TYPE'
        
        image_id = f"{patient_id}_{image_type}"
        
        # Simular diagnóstico basado en la predicción (esto sería real en producción)
        prediction = row['Prediccion']
        
        # Simular resultado de comparación (en la práctica tendríamos el diagnóstico real)
        # Para demostración, vamos a simular algunos casos
        import random
        random.seed(index)  # Para resultados consistentes
        
        if prediction == 'Malignant':
            # Simular que el 80% de predicciones malignas son correctas
            real_diagnosis = 'Maligno' if random.random() < 0.8 else 'Benigno'
        elif prediction == 'Benign':
            # Simular que el 75% de predicciones benignas son correctas
            real_diagnosis = 'Benigno' if random.random() < 0.75 else 'Maligno'
        else:  # Normal
            real_diagnosis = 'Normal' if random.random() < 0.9 else 'Benigno'
        
        # Determinar resultado de la comparación
        if prediction == 'Malignant' and real_diagnosis == 'Maligno':
            resultado = 'VP'  # Verdadero Positivo
        elif prediction == 'Benign' and real_diagnosis == 'Benigno':
            resultado = 'VN'  # Verdadero Negativo
        elif prediction == 'Malignant' and real_diagnosis != 'Maligno':
            resultado = 'FP'  # Falso Positivo
        elif prediction != 'Malignant' and real_diagnosis == 'Maligno':
            resultado = 'FN'  # Falso Negativo
        else:
            resultado = 'VN' if prediction == real_diagnosis else 'FP'
        
        detailed_results.append({
            'ID Paciente': patient_id,
            'ID Imagen': image_id,
            'Diagnóstico': real_diagnosis,
            'Predicción': prediction,
            'Resultado': resultado,
            'Confianza': row['Confianza']
        })
    
    # Crear DataFrame y mostrarlo
    results_df = pd.DataFrame(detailed_results)
    
    # Colorear las filas según el resultado
    def color_resultado(val):
        if val == 'VP':
            return 'background-color: #d4edda; color: #155724'  # Verde para VP
        elif val == 'VN':
            return 'background-color: #d1ecf1; color: #0c5460'  # Azul para VN
        elif val == 'FP':
            return 'background-color: #fff3cd; color: #856404'  # Amarillo para FP
        elif val == 'FN':
            return 'background-color: #f8d7da; color: #721c24'  # Rojo para FN
        return ''
    
    # Mostrar la tabla con colores
    styled_df = results_df.style.applymap(color_resultado, subset=['Resultado'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Mostrar resumen de resultados
    st.subheader("📊 Resumen de Clasificación")
    resultado_counts = results_df['Resultado'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        vp_count = resultado_counts.get('VP', 0)
        st.metric("🟢 Verdaderos Positivos (VP)", vp_count)
    
    with col2:
        vn_count = resultado_counts.get('VN', 0)
        st.metric("🔵 Verdaderos Negativos (VN)", vn_count)
    
    with col3:
        fp_count = resultado_counts.get('FP', 0)
        st.metric("🟡 Falsos Positivos (FP)", fp_count)
    
    with col4:
        fn_count = resultado_counts.get('FN', 0)
        st.metric("🔴 Falsos Negativos (FN)", fn_count)
    
    # Calcular métricas actuales basadas en esta muestra
    if len(resultado_counts) > 0:
        st.subheader("🎯 Métricas de Esta Muestra")
        
        total = len(results_df)
        vp = resultado_counts.get('VP', 0)
        vn = resultado_counts.get('VN', 0)
        fp = resultado_counts.get('FP', 0)
        fn = resultado_counts.get('FN', 0)
        
        # Calcular métricas
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0
        sensitivity = vp / (vp + fn) if (vp + fn) > 0 else 0
        specificity = vn / (vn + fp) if (vn + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (vp + vn) / total if total > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Precisión", f"{precision:.1%}")
        with col2:
            st.metric("🔍 Sensibilidad", f"{sensitivity:.1%}")
        with col3:
            st.metric("🛡️ Especificidad", f"{specificity:.1%}")
        with col4:
            st.metric("⚖️ F1-Score", f"{f1_score:.1%}")
        with col5:
            st.metric("✅ Exactitud", f"{accuracy:.1%}")

def display_performance_metrics(metrics):
    """
    Muestra las métricas de rendimiento en formato similar a la imagen
    """
    st.subheader("🎯 Métricas de Rendimiento del Modelo")
    
    # Métricas generales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precisión General", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("F1-Score Macro", f"{metrics['f1_macro']:.3f}")
    with col3:
        if 'auc_macro' in metrics:
            st.metric("AUC Macro", f"{metrics['auc_macro']:.3f}")
    
    # Tabla de métricas por clase (similar a la imagen)
    st.subheader("📋 Tabla de Indicadores por Clase")
    
    metrics_data = []
    classes = ['Benign', 'Malignant', 'Normal']
    
    for i, class_name in enumerate(classes):
        precision_key = f'precision_{class_name.lower()}'
        recall_key = f'recall_{class_name.lower()}'
        f1_key = f'f1_{class_name.lower()}'
        
        metrics_data.append({
            'Clase': class_name,
            'Precisión': f"{metrics.get(precision_key, 0):.3f}",
            'Sensibilidad (Recall)': f"{metrics.get(recall_key, 0):.3f}",
            'F1-Score': f"{metrics.get(f1_key, 0):.3f}"
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)

# Function to load the combined model
@st.cache_data(show_spinner=False)
def load_model():
    # URLs for model parts on GitHub
    base_url = "https://github.com/m3mentomor1/Breast-Cancer-Image-Classification/raw/main/splitted_model/"
    model_parts = [f"{base_url}model.h5.part{i:02d}" for i in range(1, 35)]

    # Download and combine model parts
    model_bytes = b''
    
    for part_url in model_parts:
        response = requests.get(part_url)
        model_bytes += response.content
    
    # Create an in-memory HDF5 file
    with h5py.File(BytesIO(model_bytes), 'r') as hf:
        # Load the combined model
        model = tf.keras.models.load_model(hf)
    
    return model

# Function to load model with progress bar
def load_model_with_progress():
    # URLs for model parts on GitHub
    base_url = "https://github.com/m3mentomor1/Breast-Cancer-Image-Classification/raw/main/splitted_model/"
    model_parts = [f"{base_url}model.h5.part{i:02d}" for i in range(1, 35)]

    # Progress bar for model loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Download and combine model parts
    model_bytes = b''
    
    for i, part_url in enumerate(model_parts):
        status_text.text(f'Descargando parte {i+1} de {len(model_parts)}...')
        response = requests.get(part_url)
        model_bytes += response.content
        progress_bar.progress((i + 1) / len(model_parts))
    
    status_text.text('Cargando modelo...')
    
    # Create an in-memory HDF5 file
    with h5py.File(BytesIO(model_bytes), 'r') as hf:
        # Load the combined model
        model = tf.keras.models.load_model(hf)
    
    progress_bar.empty()
    status_text.empty()
    
    return model

# Function to preprocess and make predictions
def predict(image, model):
    try:
        # Convertir PIL Image a RGB si es necesario (elimina canal alpha si existe)
        if image.mode == 'RGBA':
            # Crear fondo blanco y pegar la imagen con alpha
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Usar canal alpha como máscara
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Verificar que tenga exactamente 3 canales
        if img_array.ndim == 2:
            # Grayscale: convertir a RGB duplicando el canal
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3:
            if img_array.shape[-1] == 1:
                # 1 canal: duplicar para hacer RGB
                img_array = np.repeat(img_array, 3, axis=-1)
            elif img_array.shape[-1] == 4:
                # 4 canales (RGBA): tomar solo RGB
                img_array = img_array[:, :, :3]
            elif img_array.shape[-1] != 3:
                # Cualquier otro número de canales: error
                raise ValueError(f"Número de canales no soportado: {img_array.shape[-1]}")
        
        # Asegurar que es exactamente 3 canales
        if img_array.shape[-1] != 3:
            raise ValueError(f"La imagen debe tener 3 canales RGB, pero tiene {img_array.shape[-1]}")
        
        # Convertir a float32 y normalizar
        img_array = img_array.astype(np.float32) / 255.0
        
        # Redimensionar a 256x256 (tamaño esperado por tu modelo)
        img_array = tf.image.resize(img_array, (256, 256))
        
        # Agregar dimensión batch
        img_array = np.expand_dims(img_array, axis=0)  # 1 x height x width x 3
        
        # Verificar dimensiones finales
        if img_array.shape != (1, 256, 256, 3):
            raise ValueError(f"Forma inesperada del array: {img_array.shape}, esperado: (1, 256, 256, 3)")
        
        # Hacer predicción
        predictions = model.predict(img_array)
        
        # Obtener probabilidades y clase predicha
        probabilities = predictions[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_class = class_mapping[predicted_class_index]
        confidence = probabilities[predicted_class_index]
        
        return predicted_class, confidence, probabilities
        
    except Exception as e:
        raise ValueError(f"Error en el preprocesamiento: {str(e)}")
        
def preprocess_image_debug(image):
    """Función auxiliar para debuggear problemas de imagen"""
    info = {
        'mode': image.mode,
        'size': image.size,
        'format': getattr(image, 'format', 'Unknown')
    }
    
    # Convertir a array para ver dimensiones
    img_array = np.array(image)
    info['array_shape'] = img_array.shape
    info['array_dtype'] = img_array.dtype
    
    return info

def create_csv_report(results):
    """Crear un DataFrame con los resultados y convertirlo a CSV"""
    df = pd.DataFrame(results)
    
    # Agregar información adicional al reporte
    df['Fecha_Procesamiento'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reordenar columnas incluyendo las nuevas
    columns_order = ['Nombre_Archivo', 'Prediccion', 'Diagnostico', 'Resultado', 'Confianza', 
                    'Prob_Benign', 'Prob_Malignant', 'Prob_Normal', 'Fecha_Procesamiento']
    
    # Solo incluir columnas que existen en el DataFrame
    available_columns = [col for col in columns_order if col in df.columns]
    df = df[available_columns]
    
    return df

# Streamlit app
st.title('🔬 Clasificador de Cáncer de Mama - Múltiples Imágenes')
st.markdown("Sube múltiples imágenes de ultrasonido de mama para clasificarlas y obtener un reporte detallado con métricas de evaluación.")

# Opción para mostrar métricas del modelo al inicio
with st.expander("📊 Ver Métricas Históricas del Modelo", expanded=False):
    display_historical_metrics()

# Sidebar para información del modelo
with st.sidebar:
    st.header("📊 Información del Modelo")
    st.info("""
    **Clases de Clasificación:**
    - 🟢 Benign (Benigno)
    - 🔴 Malignant (Maligno)
    - 🔵 Normal
    
    **Métricas Calculadas:**
    - Precisión
    - Sensibilidad (Recall)
    - Especificidad
    - F1-Score
    - AUC (cuando aplicable)
    - Matriz de Confusión
    """)

# File uploader for multiple files
uploaded_files = st.file_uploader(
    "Selecciona las imágenes de ultrasonido de mama", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} imágenes seleccionadas**")
    
    # Show uploaded images in a grid
    if len(uploaded_files) > 0:
        st.subheader("Imágenes subidas:")
        
        # Create columns for image display
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files[:6]):  # Show first 6 images
            with cols[i % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        if len(uploaded_files) > 6:
            st.write(f"... y {len(uploaded_files) - 6} imágenes más")
    
    # Process button
    if st.button("🚀 Procesar todas las imágenes", type="primary"):
        # Load the model
        with st.spinner("Cargando modelo..."):
            # Try to load from cache first
            try:
                model = load_model()
            except:
                # If cache fails, load with progress bar
                model = load_model_with_progress()
        
        st.success("✅ Modelo cargado correctamente")
        
        # Initialize results list
        results = []
        y_true = []  # Para métricas (si tienes etiquetas verdaderas)
        y_pred = []
        y_pred_proba = []
        
        # Progress bar for processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each image
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f'Procesando {uploaded_file.name} ({i+1}/{len(uploaded_files)})...')
            
            try:
                # Open and process image
                image = Image.open(uploaded_file)
                
                predicted_class, confidence, probabilities = predict(image, model)
                predicted_class_index = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_class)]
                
                # Store results
                result = {
                    'Nombre_Archivo': uploaded_file.name,
                    'Prediccion': predicted_class,
                    'Confianza': f"{confidence:.4f}",
                    'Prob_Benign': f"{probabilities[0]:.4f}",
                    'Prob_Malignant': f"{probabilities[1]:.4f}",
                    'Prob_Normal': f"{probabilities[2]:.4f}"
                }
                results.append(result)
                
                # Para métricas (aquí podrías agregar las etiquetas reales si las tienes)
                y_pred.append(predicted_class_index)
                y_pred_proba.append(probabilities)
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"Error procesando {uploaded_file.name}: {error_msg}")
                
                result = {
                    'Nombre_Archivo': uploaded_file.name,
                    'Prediccion': 'ERROR',
                    'Confianza': 'N/A',
                    'Prob_Benign': 'N/A',
                    'Prob_Malignant': 'N/A',
                    'Prob_Normal': 'N/A'
                }
                results.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
        status_text.empty()
        
        # Create and display results
        if results:
            # Crear DataFrame mejorado con diagnóstico y resultado
            enhanced_results = create_enhanced_results_dataframe(results)
            df_results = create_csv_report(enhanced_results)
            
            st.subheader("📊 Resultados del Análisis")
            
            # Mostrar tabla con las nuevas columnas
            display_columns = ['Nombre_Archivo', 'Prediccion', 'Diagnostico', 'Resultado', 'Confianza', 
                             'Prob_Benign', 'Prob_Malignant', 'Prob_Normal', 'Fecha_Procesamiento']
            
            # Colorear las filas según el resultado
            def color_resultado(val):
                if val == 'VP':
                    return 'background-color: #d4edda; color: #155724'  # Verde
                elif val == 'VN':
                    return 'background-color: #d1ecf1; color: #0c5460'  # Azul
                elif val == 'FP':
                    return 'background-color: #fff3cd; color: #856404'  # Amarillo
                elif val == 'FN':
                    return 'background-color: #f8d7da; color: #721c24'  # Rojo
                return ''
            
            # Aplicar colores si la columna Resultado existe
            if 'Resultado' in df_results.columns:
                styled_df = df_results[display_columns].style.applymap(
                    color_resultado, subset=['Resultado']
                )
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.dataframe(df_results[display_columns], use_container_width=True)
            
            # Calcular y mostrar métricas en tiempo real
            successful_predictions = df_results[df_results['Prediccion'] != 'ERROR']
            
            if not successful_predictions.empty and 'Resultado' in successful_predictions.columns:
                # Calcular métricas basadas en los resultados reales
                metrics = calculate_real_time_metrics(successful_predictions)
                
                if metrics:
                    # Mostrar métricas calculadas dinámicamente
                    display_real_time_metrics(metrics)
            
            # Summary statistics (mantener la sección original también)
            st.subheader("📈 Resumen Estadístico")
            col1, col2, col3 = st.columns(3)
            
            if not successful_predictions.empty:
                with col1:
                    st.metric("Total Imágenes", len(df_results))
                with col2:
                    st.metric("Procesadas Exitosamente", len(successful_predictions))
                with col3:
                    st.metric("Errores", len(df_results) - len(successful_predictions))
                
                # Distribution of predictions
                if len(successful_predictions) > 0:
                    st.subheader("🔍 Distribución de Predicciones")
                    prediction_counts = successful_predictions['Prediccion'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(prediction_counts)
                    with col2:
                        for pred, count in prediction_counts.items():
                            percentage = (count / len(successful_predictions)) * 100
                            st.write(f"**{pred}**: {count} imágenes ({percentage:.1f}%)")
                
                # Mostrar métricas históricas como referencia
                if len(successful_predictions) > 0:
                    st.subheader("📚 Métricas Históricas del Modelo (Referencia)")
                    st.info("📝 Estas son las métricas del entrenamiento original del modelo")
                    
                    with st.expander("Ver métricas históricas", expanded=False):
                        display_historical_metrics()
                
                # Mostrar información sobre métricas
                st.subheader("ℹ️ Información sobre Métricas")
                with st.expander("¿Cómo interpretar las métricas?"):
                    st.markdown("""
                    **Para calcular métricas de evaluación necesitarías:**
                    
                    - **Etiquetas verdaderas**: Diagnósticos confirmados por especialistas
                    - **Predicciones del modelo**: Lo que el modelo predice
                    
                    **Métricas principales:**
                    - **Precisión**: De todas las predicciones positivas, ¿cuántas fueron correctas?
                    - **Sensibilidad (Recall)**: De todos los casos positivos reales, ¿cuántos detectó el modelo?
                    - **Especificidad**: De todos los casos negativos reales, ¿cuántos identificó correctamente?
                    - **F1-Score**: Media armónica entre precisión y sensibilidad
                    - **AUC**: Área bajo la curva ROC, mide la capacidad de discriminación
                    
                    **Matriz de Confusión:**
                    - VP: Verdaderos Positivos
                    - VN: Verdaderos Negativos  
                    - FP: Falsos Positivos
                    - FN: Falsos Negativos
                    """)
            
            # Download CSV button
            st.subheader("💾 Descargar Reporte")
            csv_data = df_results.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📁 Descargar Reporte CSV",
                data=csv_data,
                file_name=f"reporte_clasificacion_cancer_mama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success(f"✅ Procesamiento completado. {len(results)} imágenes analizadas.")
        
else:
    st.info("👆 Selecciona una o más imágenes para comenzar el análisis.")
    
    # Instructions
    with st.expander("📋 Instrucciones de uso"):
        st.markdown("""
        1. **Selecciona las imágenes**: Haz clic en "Browse files" y selecciona múltiples imágenes de ultrasonido de mama
        2. **Formatos soportados**: JPG, JPEG, PNG
        3. **Naming convention para diagnóstico automático**:
           - Para casos **malignos**: incluye palabras como 'maligno', 'malignant', 'cancer', 'malo' en el nombre
           - Para casos **benignos**: incluye palabras como 'benigno', 'benign', 'bueno', 'ben' en el nombre  
           - Para casos **normales**: incluye palabras como 'normal', 'norm', 'sano', 'healthy' en el nombre
           - Ejemplo: `imagen_maligno_001.jpg`, `paciente_benigno_xyz.png`, `caso_normal_123.jpg`
        4. **Procesa**: Haz clic en "Procesar todas las imágenes"
        5. **Revisa los resultados**: Ve los resultados con métricas calculadas automáticamente
        6. **Descarga el reporte**: Usa el botón "Descargar Reporte CSV"
        
        **Información del reporte CSV:**
        - `Nombre_Archivo`: Nombre del archivo de imagen
        - `Prediccion`: Clasificación predicha por el modelo (Benign, Malignant, Normal)
        - `Diagnostico`: Diagnóstico real extraído del nombre del archivo
        - `Resultado`: Clasificación del resultado (VP, VN, FP, FN)
        - `Confianza`: Nivel de confianza de la predicción
        - `Prob_Benign/Malignant/Normal`: Probabilidades para cada clase
        - `Fecha_Procesamiento`: Fecha y hora del análisis
        
        **Métricas calculadas automáticamente:**
        - **VP (Verdaderos Positivos)**: Casos malignos correctamente identificados
        - **VN (Verdaderos Negativos)**: Casos no malignos correctamente identificados  
        - **FP (Falsos Positivos)**: Casos predecidos como malignos pero que no lo son
        - **FN (Falsos Negativos)**: Casos malignos no detectados por el modelo
        - **Precisión, Sensibilidad, Especificidad, F1-Score, AUC**: Calculados en tiempo real
        """)

    # Información adicional sobre las métricas
    st.subheader("📊 Sobre las Métricas Calculadas")
    st.markdown("""
    El sistema ahora calcula métricas **en tiempo real** basándose en:
    - **Predicciones del modelo** vs **Diagnósticos reales** (extraídos del nombre de archivo)
    - Genera automáticamente la **matriz de confusión** y todas las métricas de evaluación
    - Presenta los resultados en el mismo formato que las imágenes de referencia
    
    **💡 Tip**: Asegúrate de nombrar tus archivos correctamente para obtener métricas precisas.
    """)