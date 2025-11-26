
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from app.config import Config

class MetricsVisualizer:
    def __init__(self):
        self.config = Config()
    
    def display_metrics_dashboard(self, metrics, results_df):
        """Mostrar dashboard completo de métricas"""
        st.subheader("📊 Dashboard de Métricas")
        
        # Tabla de totales
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📊 Matriz de Confusión - Resultados Actuales**")
            self._show_confusion_table(metrics)
        
        with col2:
            st.markdown("**🎯 Indicadores de Rendimiento**")
            self._show_metrics_table(metrics)
        
        # Gráficos circulares de progreso
        st.markdown("**📈 Visualización de Indicadores**")
        combined_chart = self._create_combined_metrics_chart(metrics)
        st.plotly_chart(combined_chart, use_container_width=True)
        
        # Gráfica ROC
        st.markdown("**📊 Análisis AUC - Curva ROC**")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            roc_chart = self._create_roc_curve_chart(results_df)
            if roc_chart:
                st.plotly_chart(roc_chart, use_container_width=True)
            else:
                simple_roc = self._create_simple_auc_chart(metrics['auc'])
                st.plotly_chart(simple_roc, use_container_width=True)
        
        with col2:
            self._show_auc_interpretation(metrics['auc'])
        
        # Métricas individuales
        # st.markdown("**🎯 Métricas Individuales**")
        # self._show_individual_metrics(metrics)
        
        # Conteos finales
        st.markdown("**🔢 Conteo de Resultados**")
        self._show_result_counts(metrics)
    
    def _show_confusion_table(self, metrics):
        """Mostrar tabla de matriz de confusión"""
        totals_df = pd.DataFrame({
            'Categoría': ['VP', 'VN', 'FP', 'FN', 'TOTALES'],
            'TOTAL': [
                metrics['VP'], metrics['VN'], metrics['FP'], 
                metrics['FN'], metrics['TOTAL']
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
    
    def _show_metrics_table(self, metrics):
        """Mostrar tabla de indicadores de rendimiento"""
        indicators_df = pd.DataFrame({
            'INDICADOR': ['Precisión', 'Sensibilidad', 'Especificidad', 'F1 Score', 'AUC'],
            'FÓRMULA': [
                'VP/(VP+FP)', 'VP/(VP+FN)', 'VN/(VN+FP)', 
                '2×(P×S)/(P+S)', '(S+E)/2'
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
    
    def _create_combined_metrics_chart(self, metrics):
        """Crear gráfico combinado de métricas"""
        metric_names = ['Precisión', 'Sensibilidad', 'Especificidad', 'F1-Score']
        metric_values = [
            metrics['precision'] * 100,
            metrics['sensitivity'] * 100,
            metrics['specificity'] * 100,
            metrics['f1_score'] * 100
        ]
        
        fig = make_subplots(
            rows=1, cols=4,
            specs=[[{"type": "pie"} for _ in range(4)]],
            subplot_titles=metric_names,
            horizontal_spacing=0.05
        )
        
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
        
        for i, (name, value, color) in enumerate(zip(metric_names, metric_values, colors)):
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
        
        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(t=50, b=50, l=10, r=10),
            title=dict(
                text="<b>🎯 Indicadores de Rendimiento</b>",
                x=0.5,
                font=dict(size=18)
            )
        )
        
        return fig
    
    def _create_roc_curve_chart(self, results_df):
        """Crear gráfica de curva ROC"""
        successful_results = results_df[results_df['Resultado'].isin(['VP', 'VN', 'FP', 'FN'])]
        
        if len(successful_results) == 0:
            return None
        
        y_true_binary = []
        y_scores = []
        
        for _, row in successful_results.iterrows():
            if row['Diagnostico'] == 'Malignant':
                y_true_binary.append(1)
            else:
                y_true_binary.append(0)
            
            try:
                malignant_prob = float(row['Prob_Malignant'])
                y_scores.append(malignant_prob)
            except:
                if row['Prediccion'] == 'Malignant':
                    y_scores.append(0.8)
                else:
                    y_scores.append(0.2)
        
        if len(set(y_true_binary)) < 2:
            return None
        
        try:
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
            auc_score = roc_auc_score(y_true_binary, y_scores)
        except:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'Curva ROC (AUC = {auc_score:.3f})',
            line=dict(color='#2ca02c', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Clasificador Aleatorio (AUC = 0.5)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        optimal_idx = np.argmax(tpr - fpr)
        fig.add_trace(go.Scatter(
            x=[fpr[optimal_idx]],
            y=[tpr[optimal_idx]],
            mode='markers',
            name=f'Punto Óptimo',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig.update_layout(
            title=f'<b>📈 Curva ROC - AUC = {auc_score:.3f}</b>',
            xaxis_title='Tasa de Falsos Positivos (1 - Especificidad)',
            yaxis_title='Tasa de Verdaderos Positivos (Sensibilidad)',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _create_simple_auc_chart(self, auc_value):
        """Crear gráfica simple de AUC"""
        x = np.linspace(0, 1, 100)
        
        if auc_value > 0.5:
            y = np.power(x, 1/(2*auc_value))
        else:
            y = 1 - np.power(1-x, 2*auc_value)
        
        y = np.clip(y, 0, 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=f'Curva ROC Estimada (AUC ≈ {auc_value:.3f})',
            line=dict(color='#2ca02c', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Clasificador Aleatorio (AUC = 0.5)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'<b>📈 Curva ROC Estimada - AUC ≈ {auc_value:.3f}</b>',
            xaxis_title='Tasa de Falsos Positivos',
            yaxis_title='Tasa de Verdaderos Positivos',
            width=600,
            height=500
        )
        
        return fig
    
    def _show_auc_interpretation(self, auc_value):
        """Mostrar interpretación del AUC"""
        st.markdown("**🎯 Interpretación del AUC:**")
        
        if auc_value >= 0.9:
            auc_interpretation = "🌟 Excelente"
        elif auc_value >= 0.8:
            auc_interpretation = "✅ Bueno"
        elif auc_value >= 0.7:
            auc_interpretation = "⚠️ Aceptable"
        elif auc_value >= 0.6:
            auc_interpretation = "🔴 Pobre"
        else:
            auc_interpretation = "❌ Muy Pobre"
        
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
    
    def _show_individual_metrics(self, metrics):
        """Mostrar métricas individuales"""
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_data = [
            ("Precisión", metrics['precision'], col1),
            ("Sensibilidad", metrics['sensitivity'], col2),
            ("Especificidad", metrics['specificity'], col3),
            ("F1-Score", metrics['f1_score'], col4)
        ]
        
        for name, value, column in metrics_data:
            with column:
                fig = self._create_circular_progress_chart(value, name)
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_circular_progress_chart(self, value, title):
        """Crear gráfico circular de progreso"""
        percentage = value * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[0.5], y=[0.5],
            mode='markers+text',
            marker=dict(size=1, color='rgba(0,0,0,0)'),
            text=f"<b>{percentage:.1f}%</b>",
            textfont=dict(size=24),
            textposition="middle center",
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Pie(
            values=[percentage, 100-percentage],
            hole=0.7,
            marker_colors=['#1f77b4', '#f0f0f0'],
            textinfo='none',
            hoverinfo='skip',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            height=200,
            margin=dict(t=50, b=10, l=10, r=10)
        )
        
        return fig
    
    def _show_result_counts(self, metrics):
        """Mostrar conteos de resultados"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 VP", metrics['VP'], help="Verdaderos Positivos")
        with col2:
            st.metric("🔵 VN", metrics['VN'], help="Verdaderos Negativos")
        with col3:
            st.metric("🟡 FP", metrics['FP'], help="Falsos Positivos")
        with col4:
            st.metric("🔴 FN", metrics['FN'], help="Falsos Negativos")