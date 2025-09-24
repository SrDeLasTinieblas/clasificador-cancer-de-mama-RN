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

# Class mapping
class_mapping = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal',
}

# Function to load the combined model
@st.cache(allow_output_mutation=True)
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
    # Convert image to numpy array
    img_array = np.array(image)

    # Si es grayscale (2D), agregar la dimensión de canales
    if img_array.ndim == 2:
        img_array = img_array[..., np.newaxis]  # height x width x 1

    # Si tiene solo 1 canal y tu modelo espera 3 canales, convertir a 3 canales
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)  # height x width x 3

    # Convertir a float32 y normalizar
    img_array = img_array.astype(np.float32) / 255.0

    # Redimensionar
    img_array = tf.image.resize(img_array, (256, 256))  # Ajusta según tu modelo

    # Agregar dimensión batch
    img_array = np.expand_dims(img_array, axis=0)  # 1 x height x width x channels

    # Hacer predicción
    predictions = model.predict(img_array)
    
    # Obtener probabilidades y clase predicha
    probabilities = predictions[0]
    predicted_class_index = np.argmax(probabilities)
    predicted_class = class_mapping[predicted_class_index]
    confidence = probabilities[predicted_class_index]
    
    return predicted_class, confidence, probabilities

def create_csv_report(results):
    """Crear un DataFrame con los resultados y convertirlo a CSV"""
    df = pd.DataFrame(results)
    
    # Agregar información adicional al reporte
    df['Fecha_Procesamiento'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reordenar columnas
    columns_order = ['Nombre_Archivo', 'Prediccion', 'Confianza', 'Prob_Benign', 'Prob_Malignant', 'Prob_Normal', 'Fecha_Procesamiento']
    df = df[columns_order]
    
    return df

# Streamlit app
st.title('🔬 Clasificador de Cáncer de Mama - Múltiples Imágenes')
st.markdown("Sube múltiples imágenes de ultrasonido de mama para clasificarlas y obtener un reporte detallado.")

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
                
            except Exception as e:
                st.error(f"Error procesando {uploaded_file.name}: {str(e)}")
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
            st.subheader("📊 Resultados del Análisis")
            
            # Create DataFrame
            df_results = create_csv_report(results)
            
            # Display results table
            st.dataframe(df_results, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Resumen Estadístico")
            col1, col2, col3 = st.columns(3)
            
            successful_predictions = df_results[df_results['Prediccion'] != 'ERROR']
            
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
        3. **Procesa**: Haz clic en "Procesar todas las imágenes"
        4. **Revisa los resultados**: Ve los resultados en la tabla y el resumen estadístico
        5. **Descarga el reporte**: Usa el botón "Descargar Reporte CSV" para obtener un archivo con todos los resultados
        
        **Información del reporte CSV:**
        - `Nombre_Archivo`: Nombre del archivo de imagen
        - `Prediccion`: Clasificación predicha (Benign, Malignant, Normal)
        - `Confianza`: Nivel de confianza de la predicción
        - `Prob_Benign/Malignant/Normal`: Probabilidades para cada clase
        - `Fecha_Procesamiento`: Fecha y hora del análisis
        """)