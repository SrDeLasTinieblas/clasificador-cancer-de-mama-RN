import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

def create_circular_progress_chart(value, title, color_scheme="blue"):
    """
    Crea un gráfico circular de progreso para mostrar porcentajes
    """
    # Convertir a porcentaje si está en decimal
    if value <= 1:
        percentage = value * 100
    else:
        percentage = value
    
    # Definir colores según el esquema
    color_schemes = {
        "blue": {"main": "#1f77b4", "bg": "#e6f2ff"},
        "green": {"main": "#2ca02c", "bg": "#e6ffe6"},
        "orange": {"main": "#ff7f0e", "bg": "#fff2e6"},
        "red": {"main": "#d62728", "bg": "#ffe6e6"},
        "purple": {"main": "#9467bd", "bg": "#f3e6ff"}
    }
    
    colors = color_schemes.get(color_scheme, color_schemes["blue"])
    
    # Crear el gráfico circular
    fig = go.Figure()
    
    # Agregar el arco de progreso
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode='markers+text',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        text=f"<b>{percentage:.1f}%</b>",
        textfont=dict(size=24, color=colors["main"]),
        textposition="middle center",
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Crear el gráfico de dona
    fig.add_trace(go.Pie(
        values=[percentage, 100-percentage],
        hole=0.7,
        marker_colors=[colors["main"], colors["bg"]],
        textinfo='none',
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Configurar el layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=16, color=colors["main"])
        ),
        height=200,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=title,
                x=0.5, y=-0.1,
                xref="paper", yref="paper",
                xanchor="center", yanchor="top",
                font=dict(size=12, color="#666666"),
                showarrow=False
            )
        ]
    )
    
    return fig

def create_roc_curve_chart(results_df):
    """
    Crea una gráfica de curva ROC para visualizar el AUC
    """
    # Filtrar solo resultados exitosos
    successful_results = results_df[results_df['Resultado'].isin(['VP', 'VN', 'FP', 'FN'])]
    
    if len(successful_results) == 0:
        return None
    
    # Convertir diagnósticos y predicciones a valores binarios (Maligno vs No-Maligno)
    y_true_binary = []
    y_scores = []
    
    for _, row in successful_results.iterrows():
        # True label: 1 si es maligno, 0 si no
        if row['Diagnostico'] == 'Malignant':
            y_true_binary.append(1)
        else:
            y_true_binary.append(0)
        
        # Score: probabilidad de ser maligno
        try:
            malignant_prob = float(row['Prob_Malignant'])
            y_scores.append(malignant_prob)
        except:
            # Si hay error, usar valor por defecto basado en predicción
            if row['Prediccion'] == 'Malignant':
                y_scores.append(0.8)
            else:
                y_scores.append(0.2)
    
    if len(set(y_true_binary)) < 2:
        # No hay suficiente variabilidad para ROC
        return create_simple_auc_chart(calculate_simple_auc(successful_results))
    
    # Calcular curva ROC
    try:
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
        auc_score = roc_auc_score(y_true_binary, y_scores)
    except:
        return create_simple_auc_chart(calculate_simple_auc(successful_results))
    
    # Crear la gráfica
    fig = go.Figure()
    
    # Línea de la curva ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'Curva ROC (AUC = {auc_score:.3f})',
        line=dict(color='#2ca02c', width=3),
        fill='tonexty' if len(fpr) > 0 else None,
        fillcolor='rgba(44, 160, 44, 0.1)'
    ))
    
    # Línea diagonal (clasificador aleatorio)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Clasificador Aleatorio (AUC = 0.5)',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=True
    ))
    
    # Punto óptimo (más cercano a la esquina superior izquierda)
    optimal_idx = np.argmax(tpr - fpr)
    fig.add_trace(go.Scatter(
        x=[fpr[optimal_idx]],
        y=[tpr[optimal_idx]],
        mode='markers',
        name=f'Punto Óptimo (TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f})',
        marker=dict(color='red', size=12, symbol='star'),
        showlegend=True
    ))
    
    # Configurar layout con mejor legibilidad
    fig.update_layout(
        title=dict(
            text=f'<b>📈 Curva ROC - AUC = {auc_score:.3f}</b>',
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title='Tasa de Falsos Positivos (1 - Especificidad)',
        yaxis_title='Tasa de Verdaderos Positivos (Sensibilidad)',
        width=600,
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=2,
            font=dict(size=12, color='#2c3e50')
        ),
        xaxis=dict(
            range=[0, 1], 
            constrain='domain',
            title=dict(font=dict(size=14, color='#2c3e50')),
            tickfont=dict(size=12, color='#2c3e50'),
            tickmode='linear',
            tick0=0,
            dtick=0.2
        ),
        yaxis=dict(
            range=[0, 1], 
            scaleanchor='x', 
            scaleratio=1,
            title=dict(font=dict(size=14, color='#2c3e50')),
            tickfont=dict(size=12, color='#2c3e50'),
            tickmode='linear',
            tick0=0,
            dtick=0.2
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#2c3e50')
    )
    
    # Agregar grid con mejor contraste
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1.5, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True,
        linewidth=2,
        linecolor='#2c3e50',
        mirror=True
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1.5, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True,
        linewidth=2,
        linecolor='#2c3e50',
        mirror=True
    )
    
    return fig

def create_simple_auc_chart(auc_value):
    """
    Crea una gráfica simple de AUC cuando no se puede calcular ROC
    """
    fig = go.Figure()
    
    # Crear una curva ROC simulada basada en el AUC
    x = np.linspace(0, 1, 100)
    
    # Aproximar una curva que tenga el AUC deseado
    if auc_value > 0.5:
        # Curva convexa para AUC > 0.5
        y = np.power(x, 1/(2*auc_value))
    else:
        # Curva cóncava para AUC < 0.5
        y = 1 - np.power(1-x, 2*auc_value)
    
    # Asegurar que esté entre 0 y 1
    y = np.clip(y, 0, 1)
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=f'Curva ROC Estimada (AUC ≈ {auc_value:.3f})',
        line=dict(color='#2ca02c', width=3),
        fill='tonexty',
        fillcolor='rgba(44, 160, 44, 0.1)'
    ))
    
    # Línea diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Clasificador Aleatorio (AUC = 0.5)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>📈 Curva ROC Estimada - AUC ≈ {auc_value:.3f}</b>',
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title='Tasa de Falsos Positivos (1 - Especificidad)',
        yaxis_title='Tasa de Verdaderos Positivos (Sensibilidad)',
        width=600,
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=2,
            font=dict(size=12, color='#2c3e50')
        ),
        xaxis=dict(
            range=[0, 1],
            title=dict(font=dict(size=14, color='#2c3e50')),
            tickfont=dict(size=12, color='#2c3e50'),
            tickmode='linear',
            tick0=0,
            dtick=0.2
        ),
        yaxis=dict(
            range=[0, 1],
            title=dict(font=dict(size=14, color='#2c3e50')),
            tickfont=dict(size=12, color='#2c3e50'),
            tickmode='linear',
            tick0=0,
            dtick=0.2
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#2c3e50')
    )
    
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1.5, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True,
        linewidth=2,
        linecolor='#2c3e50',
        mirror=True
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1.5, 
        gridcolor='rgba(128,128,128,0.3)',
        showline=True,
        linewidth=2,
        linecolor='#2c3e50',
        mirror=True
    )
    
    return fig

def calculate_simple_auc(results_df):
    """
    Calcula un AUC simplificado basado en VP, VN, FP, FN
    """
    counts = results_df['Resultado'].value_counts()
    vp = counts.get('VP', 0)
    vn = counts.get('VN', 0) 
    fp = counts.get('FP', 0)
    fn = counts.get('FN', 0)
    
    if (vp + fn) == 0 or (vn + fp) == 0:
        return 0.5
    
    sensitivity = vp / (vp + fn)
    specificity = vn / (vn + fp)
    
    # Aproximación simple del AUC
    return (sensitivity + specificity) / 2

def create_combined_metrics_chart(metrics):
    """
    Crea un gráfico combinado con todos los indicadores de rendimiento
    """
    # Preparar datos
    metric_names = ['Precisión', 'Sensibilidad', 'Especificidad', 'F1-Score']
    metric_values = [
        metrics['precision'] * 100,
        metrics['sensitivity'] * 100,
        metrics['specificity'] * 100,
        metrics['f1_score'] * 100
    ]
    
    # Crear subplots para los gráficos circulares (4 en lugar de 5)
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "pie"} for _ in range(4)]],
        subplot_titles=metric_names,
        horizontal_spacing=0.05
    )
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    for i, (name, value, color) in enumerate(zip(metric_names, metric_values, colors)):
        # Agregar gráfico de dona
        fig.add_trace(
            go.Pie(
                values=[value, 100-value],
                hole=0.7,
                marker_colors=[color, '#f0f0f0'],
                textinfo='none',
                hoverinfo='skip',
                showlegend=False,
                name=name
            ),
            row=1, col=i+1
        )
        
        # Agregar texto en el centro
        fig.add_annotation(
            text=f"<b>{value:.1f}%</b>",
            x=(i * 0.25) + 0.125,
            y=0.5,
            xref="paper",
            yref="paper",
            font=dict(size=16, color=color),
            showarrow=False,
            xanchor="center",
            yanchor="middle"
        )
    
    # Configurar layout
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(t=50, b=50, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="<b>🎯 Indicadores de Rendimiento - Calculados</b>",
            x=0.5,
            font=dict(size=18)
        )
    )
    
    return fig

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
    
    # *** NUEVA SECCIÓN: Gráficos Circulares de Progreso ***
    st.markdown("**📈 Visualización de Indicadores de Rendimiento**")
    
    # Crear gráfico combinado (ahora sin AUC)
    combined_chart = create_combined_metrics_chart(metrics)
    st.plotly_chart(combined_chart, use_container_width=True)
    
    # *** NUEVA SECCIÓN: Gráfica ROC para AUC ***
    st.markdown("**📊 Análisis AUC - Curva ROC**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Crear y mostrar gráfica ROC
        current_data = st.session_state.get('current_results_df')
        if current_data is not None:
            roc_chart = create_roc_curve_chart(current_data)
            if roc_chart:
                st.plotly_chart(roc_chart, use_container_width=True)
            else:
                # Si no se puede crear ROC real, crear una estimada
                simple_roc = create_simple_auc_chart(metrics['auc'])
                st.plotly_chart(simple_roc, use_container_width=True)
        else:
            # Crear AUC estimado
            simple_roc = create_simple_auc_chart(metrics['auc'])
            st.plotly_chart(simple_roc, use_container_width=True)
    
    with col2:
        # Información sobre AUC
        st.markdown("**🎯 Interpretación del AUC:**")
        auc_value = metrics['auc']
        
        if auc_value >= 0.9:
            auc_interpretation = "🌟 Excelente"
            auc_color = "green"
        elif auc_value >= 0.8:
            auc_interpretation = "✅ Bueno"
            auc_color = "blue"
        elif auc_value >= 0.7:
            auc_interpretation = "⚠️ Aceptable"
            auc_color = "orange"
        elif auc_value >= 0.6:
            auc_interpretation = "🔴 Pobre"
            auc_color = "red"
        else:
            auc_interpretation = "❌ Muy Pobre"
            auc_color = "red"
        
        st.metric(
            label="📊 AUC Score", 
            value=f"{auc_value:.3f}",
            help="Área bajo la curva ROC"
        )
        
        st.markdown(f"**Clasificación:** {auc_interpretation}")
        
        st.markdown("""
        **Rangos de AUC:**
        - 0.9-1.0: Excelente
        - 0.8-0.9: Bueno  
        - 0.7-0.8: Aceptable
        - 0.6-0.7: Pobre
        - 0.5-0.6: Muy Pobre
        - 0.5: Aleatorio
        """)
    
    # También mostrar gráficos individuales en columnas (sin AUC)
    st.markdown("**🎯 Métricas Individuales**")
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("Precisión", metrics['precision'], "blue", col1),
        ("Sensibilidad", metrics['sensitivity'], "green", col2),
        ("Especificidad", metrics['specificity'], "orange", col3),
        ("F1-Score", metrics['f1_score'], "red", col4)
    ]
    
    for name, value, color, column in metrics_data:
        with column:
            fig = create_circular_progress_chart(value, name, color)
            st.plotly_chart(fig, use_container_width=True)
    
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

# CONFIGURACIÓN DEL MODELO - CAMBIAR ESTA RUTA
MODEL_PATH = r"D:\Empresa\Cancer de mama\RN_mama_ya_entrenado\Breast-Cancer-Image-Classification-with-DenseNet121\models\model_complete.h5"

# CÓDIGO DE DEPURACIÓN - Agregar al final del archivo
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Diagnóstico del Modelo")
st.sidebar.write(f"Ruta configurada: `{MODEL_PATH}`")
st.sidebar.write(f"¿Archivo existe?: {os.path.exists(MODEL_PATH)}")

if os.path.exists(MODEL_PATH):
    try:
        file_size = os.path.getsize(MODEL_PATH)
        st.sidebar.write(f"Tamaño del archivo: {file_size / (1024*1024):.1f} MB")
    except:
        st.sidebar.write("Error obteniendo tamaño del archivo")
else:
    st.sidebar.write("❌ El archivo no existe en esta ruta")
    
    # Verificar si la carpeta padre existe
    parent_dir = os.path.dirname(MODEL_PATH)
    st.sidebar.write(f"¿Carpeta padre existe?: {os.path.exists(parent_dir)}")
    
    if os.path.exists(parent_dir):
        try:
            files_in_dir = os.listdir(parent_dir)
            h5_files = [f for f in files_in_dir if f.endswith('.h5')]
            st.sidebar.write(f"Archivos .h5 encontrados en la carpeta: {h5_files}")
        except:
            st.sidebar.write("Error listando archivos de la carpeta")

# Limpiar cache si es necesario
if st.sidebar.button("🧹 Limpiar Cache y Recargar Modelo"):
    st.cache_resource.clear()
    st.rerun()

@st.cache_resource
def load_local_model():
    """
    Carga un modelo local desde un archivo .h5
    """
    try:
        # st.write(f"Intentando cargar modelo desde: {MODEL_PATH}")  # Debug
        model = tf.keras.models.load_model(MODEL_PATH)
        # st.write("✅ Modelo cargado exitosamente")  # Debug
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo desde {MODEL_PATH}: {str(e)}")
        return None

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

# Streamlit app
st.title('🔬 Clasificador de Cáncer de Mama - Múltiples Imágenes')
st.markdown("Sube múltiples imágenes de ultrasonido de mama para clasificarlas y obtener un reporte detallado con métricas de evaluación.")

# Cargar el modelo al inicio
@st.cache_resource
def get_model():
    return load_local_model()

model = get_model()

# Sidebar para información del modelo
with st.sidebar:
    st.header("📊 Información del Modelo")
    
    # Mostrar ruta configurada
    st.info(f"""
    **Ruta del modelo:** 
    `{MODEL_PATH}`
    
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
    
    if model is None:
        st.error("⚠️ Modelo no cargado")
        st.markdown("""
        **Para solucionarlo:**
        1. Verifica que el archivo existe
        2. Cambia MODEL_PATH en el código
        3. Reinicia la aplicación
        """)
    else:
        st.success("✅ Modelo operativo")

# Mostrar estado del modelo
if model is None:
    st.error("❌ No se pudo cargar el modelo desde la ruta especificada")
    st.info(f"📁 Ruta configurada: {MODEL_PATH}")
    st.info("💡 Verifica que el archivo existe y la ruta sea correcta")
    st.markdown("### 🔧 Posibles soluciones:")
    st.markdown("""
    1. **Verifica la ruta**: Asegúrate de que el archivo existe en la ubicación especificada
    2. **Usa barras normales**: Cambia `D:\Empresa\...` por `D:/Empresa/...` o usa raw string `r"D:\Empresa\..."`
    3. **Permisos**: Verifica que tienes permisos de lectura en esa carpeta
    4. **Modelo válido**: Asegúrate de que el archivo .h5 no esté corrupto
    """)
else:
    st.success("✅ Modelo cargado correctamente y listo para usar")
    st.info(f"📁 Modelo cargado desde: {MODEL_PATH}")

# File uploader for multiple files
uploaded_files = st.file_uploader(
    "Selecciona las imágenes de ultrasonido de mama", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    disabled=(model is None)
)

if uploaded_files and model is not None:
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
                # Almacenar en session state para usar en gráfica ROC
                st.session_state['current_results_df'] = successful_predictions
                
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
    if model is None:
        st.info("🧠 Carga un modelo local para comenzar el análisis.")
    else:
        st.info("👆 Selecciona una o más imágenes para comenzar el análisis.")
    
    # Instructions
    with st.expander("📋 Instrucciones de uso"):
        st.markdown("""
        1. **Carga el modelo**: En la barra lateral, selecciona "Cargar archivo local" y sube tu archivo .h5, o usa "Usar ruta específica" para indicar la ubicación del modelo
        2. **Selecciona las imágenes**: Haz clic en "Browse files" y selecciona múltiples imágenes de ultrasonido de mama
        3. **Formatos soportados**: JPG, JPEG, PNG
        4. **Naming convention para diagnóstico automático**:
           - Para casos **malignos**: incluye palabras como 'maligno', 'malignant', 'cancer', 'malo' en el nombre
           - Para casos **benignos**: incluye palabras como 'benigno', 'benign', 'bueno', 'ben' en el nombre  
           - Para casos **normales**: incluye palabras como 'normal', 'norm', 'sano', 'healthy' en el nombre
           - Ejemplo: `imagen_maligno_001.jpg`, `paciente_benigno_xyz.png`, `caso_normal_123.jpg`
        5. **Procesa**: Haz clic en "Procesar todas las imágenes"
        6. **Revisa los resultados**: Ve los resultados con métricas calculadas automáticamente
        7. **Descarga el reporte**: Usa el botón "Descargar Reporte CSV"
        
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
    
    **🚀 Ventajas del modelo local:**
    - ⚡ Carga instantánea (sin descargas)
    - 🔒 Mayor privacidad (procesamiento local)
    - 🏃‍♂️ Análisis más rápido
    - 📱 Funciona sin conexión a internet
    """)