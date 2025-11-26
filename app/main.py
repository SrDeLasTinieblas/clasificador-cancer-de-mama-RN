import streamlit as st
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.utils.model_utils import ModelManager
from app.utils.image_processing import ImageProcessor
from app.utils.metrics_calculator import MetricsCalculator
from app.utils.report_generator import ReportGenerator
from app.utils.visualization import MetricsVisualizer
from app.utils.auth import AuthManager
from app.config import Config

def initialize_components():
    """Inicializar componentes del sistema"""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor()
    if 'metrics_calc' not in st.session_state:
        st.session_state.metrics_calc = MetricsCalculator()
    if 'report_gen' not in st.session_state:
        st.session_state.report_gen = ReportGenerator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = MetricsVisualizer()
    
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'enhanced_results' not in st.session_state:
        st.session_state.enhanced_results = None
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None
    if 'analysis_metrics' not in st.session_state:
        st.session_state.analysis_metrics = None
    if 'successful_predictions' not in st.session_state:
        st.session_state.successful_predictions = None

def clear_analysis_results():
    """Limpiar resultados de análisis previos"""
    st.session_state.analysis_completed = False
    st.session_state.analysis_results = None
    st.session_state.enhanced_results = None
    st.session_state.df_results = None
    st.session_state.analysis_metrics = None
    st.session_state.successful_predictions = None
    if 'current_results_df' in st.session_state:
        del st.session_state.current_results_df

def main():
    st.set_page_config(
        page_title="Clasificador de Cáncer de Mama",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sistema de autenticación
    auth_manager = AuthManager()

    # Verificar si el usuario está autenticado
    if not auth_manager.is_authenticated():
        # Mostrar formulario de login
        auth_manager.show_login_form()
        return

    # Usuario autenticado - mostrar aplicación principal
    st.title('🔬 Clasificador de Cáncer de Mama - Múltiples Imágenes')
    st.markdown("Sube múltiples imágenes de ultrasonido de mama para clasificarlas y obtener un reporte detallado con métricas de evaluación.")

    # Mostrar información del usuario en la barra lateral
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**👤 Usuario:** {auth_manager.get_current_user()}")
        if st.button("🚪 Cerrar Sesión", key="logout_btn"):
            auth_manager.logout()
            st.experimental_rerun()
        st.markdown("---")

    initialize_components()
    
    model_manager = st.session_state.model_manager
    image_processor = st.session_state.image_processor
    metrics_calc = st.session_state.metrics_calc
    report_gen = st.session_state.report_gen
    visualizer = st.session_state.visualizer
    
    with st.sidebar:
        st.header("📊 Configuración del Modelo")
        
        model = model_manager.load_model_interface()
        
        if model is not None:
            st.success("✅ Modelo operativo")
            
            if st.session_state.analysis_completed:
                if st.button("🗑️ Nuevo Análisis", help="Limpiar resultados previos"):
                    clear_analysis_results()
                    st.experimental_rerun()
            
            with st.expander("ℹ️ Info del Sistema"):
                st.markdown(f"""
                **Clases de Clasificación:**
                - 🟢 Benign (Benigno)
                - 🔴 Malignant (Maligno)  
                - 🔵 Normal
                
                **Métricas Calculadas:**
                - Precisión, Sensibilidad
                - Especificidad, F1-Score
                - AUC, Matriz de Confusión
                
                **Estado del Modelo:**
                - Cargado en memoria
                - Listo para predicciones
                """)
        else:
            st.error("⚠️ Modelo no cargado")
            st.markdown("""
            **Para usar la aplicación:**
            1. Sube un modelo .h5, o
            2. Especifica una ruta válida, o
            3. Verifica la ruta por defecto
            """)
    
    if model is not None:
        if st.session_state.analysis_completed and st.session_state.df_results is not None:
            show_persistent_results()
        else:
            uploaded_files = st.file_uploader(
                "Selecciona las imágenes de ultrasonido de mama", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True,
                help="Puedes seleccionar múltiples archivos a la vez"
            )
            
            if uploaded_files:
                process_images(uploaded_files, model, image_processor, metrics_calc, report_gen, visualizer)
            else:
                show_instructions()
    else:
        show_instructions()

def process_images(uploaded_files, model, image_processor, metrics_calc, report_gen, visualizer):
    """Procesar múltiples imágenes y mostrar resultados"""
    st.write(f"**{len(uploaded_files)} imágenes seleccionadas**")
    
    if len(uploaded_files) > 0:
        st.subheader("Imágenes subidas:")
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files[:6]):
            with cols[i % 3]:
                try:
                    image = image_processor.load_image(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                except Exception as e:
                    st.error(f"Error cargando {uploaded_file.name}: {str(e)}")
        
        if len(uploaded_files) > 6:
            st.write(f"... y {len(uploaded_files) - 6} imágenes más")
    
    if st.button("🚀 Procesar todas las imágenes", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f'Procesando {uploaded_file.name} ({i+1}/{len(uploaded_files)})...')
            
            try:
                image = image_processor.load_image(uploaded_file)
                prediction_result = image_processor.predict_image(image, model)
                
                result = {
                    'Nombre_Archivo': uploaded_file.name,
                    **prediction_result
                }
                results.append(result)
                
            except Exception as e:
                st.error(f"Error procesando {uploaded_file.name}: {str(e)}")
                results.append({
                    'Nombre_Archivo': uploaded_file.name,
                    'Prediccion': 'ERROR',
                    'Confianza': 'N/A',
                    'Prob_Benign': 'N/A',
                    'Prob_Malignant': 'N/A',
                    'Prob_Normal': 'N/A',
                    'Error': str(e)
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            enhanced_results = report_gen.create_enhanced_results(results)
            df_results = report_gen.create_dataframe(enhanced_results)
            
            successful_predictions = df_results[df_results['Prediccion'] != 'ERROR']
            metrics = None
            if not successful_predictions.empty and 'Resultado' in successful_predictions.columns:
                metrics = metrics_calc.calculate_real_time_metrics(successful_predictions)
            
            st.session_state.analysis_results = results
            st.session_state.enhanced_results = enhanced_results
            st.session_state.df_results = df_results
            st.session_state.successful_predictions = successful_predictions
            st.session_state.analysis_metrics = metrics
            st.session_state.analysis_completed = True
            st.session_state.current_results_df = successful_predictions
            
            st.success(f"✅ Procesamiento completado. {len(results)} imágenes analizadas.")
            st.experimental_rerun()

def show_persistent_results():
    """Mostrar resultados persistentes del análisis"""
    st.subheader("📊 Resultados del Análisis")
    
    df_results = st.session_state.df_results
    enhanced_results = st.session_state.enhanced_results
    successful_predictions = st.session_state.successful_predictions
    metrics = st.session_state.analysis_metrics
    report_gen = st.session_state.report_gen
    visualizer = st.session_state.visualizer
    
    def color_resultado(val):
        colors_map = {
            'VP': 'background-color: #d4edda; color: #155724',
            'VN': 'background-color: #d1ecf1; color: #0c5460',
            'FP': 'background-color: #fff3cd; color: #856404',
            'FN': 'background-color: #f8d7da; color: #721c24',
            'ERROR': 'background-color: #f5f5f5; color: #6c757d'
        }
        return colors_map.get(val, '')
    
    if 'Resultado' in df_results.columns:
        styled_df = df_results.style.applymap(color_resultado, subset=['Resultado'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df_results, use_container_width=True)
    
    if metrics and not successful_predictions.empty:
        visualizer.display_metrics_dashboard(metrics, successful_predictions)
    
    show_summary_stats(df_results, successful_predictions)
    
    show_download_section_persistent(enhanced_results, df_results, report_gen)

def show_download_section_persistent(enhanced_results, df_results, report_gen):
    """Mostrar sección de descarga usando datos del session_state"""
    st.subheader("💾 Descargar Reporte")
    
    if 'excel_report_data' not in st.session_state:
        st.session_state.excel_report_data = report_gen.create_excel_report(enhanced_results)
    if 'csv_report_data' not in st.session_state:
        st.session_state.csv_report_data = df_results.to_csv(index=False, encoding='utf-8-sig')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Descargar Reporte Excel",
            data=st.session_state.excel_report_data,
            file_name=f"reporte_cancer_mama_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Reporte profesional con formato y colores"
        )

    st.info("💡 Los reportes se mantienen disponibles hasta que hagas un nuevo análisis")

def show_summary_stats(df_results, successful_predictions):
    """Mostrar resumen estadístico"""
    st.subheader("📈 Resumen Estadístico")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Imágenes", len(df_results))
    with col2:
        st.metric("Procesadas Exitosamente", len(successful_predictions))
    with col3:
        st.metric("Errores", len(df_results) - len(successful_predictions))
    
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

def show_instructions():
    """Mostrar instrucciones de uso"""
    if st.session_state.model_manager.get_current_model() is None:
        st.info("🧠 Carga un modelo para comenzar el análisis.")
    else:
        st.info("👆 Selecciona una o más imágenes para comenzar el análisis.")
    
    with st.expander("📋 Instrucciones de uso"):
        st.markdown("""
        ## 🧠 Carga del Modelo

        **Opción 1 - Subir archivo (Recomendado para modelos nuevos):**
        1. Ve a la barra lateral → "Cargar Modelo"
        2. Usa "Subir Archivo" - **Hasta 2GB soportado**
        3. Haz clic en "Cargar Modelo Subido"

        **Opción 2 - Especificar ruta (Más rápido):**
        1. Ve a la barra lateral → "Ruta del Modelo"
        2. Ingresa la ruta completa de tu modelo
        3. Haz clic en "Cargar desde Ruta"

        **Opción 3 - Descargar desde link (Para archivos partidos):**
        1. Ve a la barra lateral → "Descargar desde Link"
        2. Ingresa las URLs de las partes del modelo (separadas por comas o líneas)
        3. Haz clic en "Descargar y Unir Modelo"
        4. Las partes se descargarán, unirán y cargarán automáticamente

        **Para cambiar de modelo:**
        - Usa "Limpiar Cache y Resetear" y carga un nuevo modelo
        
        ## 📸 Procesamiento de Imágenes
        
        1. **Selecciona las imágenes**: Múltiples imágenes JPG, JPEG, PNG
        2. **Naming convention para diagnóstico automático**:
           - **Malignos**: incluye 'maligno', 'malignant', 'cancer', 'malo'
           - **Benignos**: incluye 'benigno', 'benign', 'bueno', 'ben'  
           - **Normales**: incluye 'normal', 'norm', 'sano', 'healthy'
           - Ejemplo: `imagen_maligno_001.jpg`, `caso_benigno_xyz.png`
        3. **Procesa**: Haz clic en "Procesar todas las imágenes"
        4. **Revisa los resultados**: Ve métricas calculadas automáticamente
        5. **Descarga el reporte**: Excel (profesional) o CSV (simple)
        
        ## 📊 Características del sistema:
        - **Resultados persistentes**: Los datos no se pierden al descargar reportes
        - **Múltiples descargas**: Puedes descargar los reportes varias veces
        - **Análisis completo**: Métricas calculadas automáticamente
        - **Interfaz estable**: El modelo permanece cargado durante la sesión
        
        ## 🎯 Métricas calculadas automáticamente:
        - **VP (Verdaderos Positivos)**: Casos malignos correctamente identificados
        - **VN (Verdaderos Negativos)**: Casos no malignos correctamente identificados  
        - **FP (Falsos Positivos)**: Casos predecidos como malignos pero que no lo son
        - **FN (Falsos Negativos)**: Casos malignos no detectados por el modelo
        - **Precisión, Sensibilidad, Especificidad, F1-Score, AUC**: Calculados en tiempo real
        """)

if __name__ == "__main__":
    main()