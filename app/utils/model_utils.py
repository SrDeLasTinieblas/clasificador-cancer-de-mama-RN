import streamlit as st
import tensorflow as tf
import os
import tempfile
from pathlib import Path
from app.config import Config
from app.utils.file_downloader import FileDownloader

class ModelManager:
    def __init__(self):
        self.config = Config()
        self.file_downloader = FileDownloader()

        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = None
        if 'model_path' not in st.session_state:
            st.session_state.model_path = None
    
    @st.cache_resource
    def _load_model_from_path(_self, model_path):
        """Cargar modelo desde una ruta específica (versión interna con cache)"""
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error cargando modelo: {str(e)}")
            return None
    
    def load_model_from_path(self, model_path):
        """Cargar modelo desde ruta y guardarlo en session_state"""
        try:
            file_size = os.path.getsize(model_path)
            st.info(f"Cargando modelo ({file_size / (1024*1024):.1f} MB)...")
            
            model = self._load_model_from_path(model_path)
            
            if model:
                st.session_state.model_loaded = model
                st.session_state.model_path = model_path
                st.session_state.model_info = {
                    'source': 'path',
                    'path': model_path,
                    'size_mb': file_size / (1024*1024),
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape,
                    'total_params': model.count_params()
                }
                st.success("Modelo cargado exitosamente")
            
            return model
        except Exception as e:
            st.error(f"Error cargando modelo: {str(e)}")
            return None
    
    def load_model_from_upload(self, uploaded_file):
        """Cargar modelo desde archivo subido y mantenerlo en session_state"""
        if uploaded_file is not None:
            try:
                file_size = uploaded_file.size
                st.info(f"Procesando archivo subido: {uploaded_file.name} ({file_size / (1024*1024):.1f} MB)")
                
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    chunk_size = 1024 * 1024  # 1MB chunks
                    bytes_written = 0
                    
                    uploaded_file.seek(0)
                    
                    while True:
                        chunk = uploaded_file.read(chunk_size)
                        if not chunk:
                            break
                        
                        tmp_file.write(chunk)
                        bytes_written += len(chunk)
                        
                        progress = min(bytes_written / file_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Escribiendo archivo: {progress:.1%}")
                    
                    tmp_file_path = tmp_file.name
                
                progress_bar.empty()
                status_text.empty()
                
                st.info("Cargando modelo en memoria...")
                model = self._load_model_from_path(tmp_file_path)
                
                if model:
                    permanent_path = self.config.MODELS_DIR / f"uploaded_{uploaded_file.name}"
                    permanent_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save(str(permanent_path))
                    
                    st.session_state.model_loaded = model
                    st.session_state.model_path = str(permanent_path)
                    st.session_state.model_info = {
                        'source': 'upload',
                        'original_name': uploaded_file.name,
                        'path': str(permanent_path),
                        'size_mb': file_size / (1024*1024),
                        'input_shape': model.input_shape,
                        'output_shape': model.output_shape,
                        'total_params': model.count_params()
                    }
                    
                    st.success(f"Modelo guardado en: {permanent_path}")
                
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                
                return model
                
            except Exception as e:
                st.error(f"Error procesando archivo subido: {str(e)}")
                return None
        return None
    
    def get_current_model(self):
        """Obtener el modelo actual del session_state"""
        return st.session_state.model_loaded
    
    def clear_model(self):
        """Limpiar modelo del session_state y cache"""
        st.session_state.model_loaded = None
        st.session_state.model_info = None
        st.session_state.model_path = None
        st.cache_resource.clear()
    
    def load_model_interface(self):
        """Interface de Streamlit para cargar modelos con estado persistente"""
        st.subheader("Cargar Modelo")
        
        current_model = self.get_current_model()
        if current_model is not None:
            st.success("Modelo ya cargado y listo para usar")
            if st.session_state.model_info:
                info = st.session_state.model_info
                st.info(f"Modelo activo: {info.get('original_name', 'modelo')} ({info.get('size_mb', 0):.1f} MB)")
        
        st.info("Límite de archivo configurado: 2GB (2048MB)")
        
        st.markdown("**Opción 1: Subir Archivo**")
        
        with st.expander("Instrucciones para archivos grandes"):
            st.markdown("""
            **Para modelos grandes (>200MB):**
            1. Asegúrate que el archivo `.streamlit/config.toml` esté configurado
            2. El proceso puede tomar varios minutos
            3. No cierres la pestaña durante la carga
            4. Se mostrará una barra de progreso
            
            **Si sigues teniendo problemas:**
            - Usa la "Opción 2: Ruta del Modelo" (más rápido)
            - Verifica que tienes suficiente RAM disponible
            - Reinicia la aplicación si es necesario
            """)
        
        uploaded_file = st.file_uploader(
            "Sube tu modelo (.h5)", 
            type=["h5"],
            help="Archivos hasta 2GB soportados",
            key="model_uploader"
        )
        
        if uploaded_file:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Archivo", uploaded_file.name)
            with col2:
                st.metric("Tamaño", f"{uploaded_file.size / (1024*1024):.1f} MB")
            with col3:
                st.metric("Tipo", uploaded_file.type)
            
            if st.button("Cargar Modelo Subido", type="primary", key="btn_upload"):
                with st.spinner(f"Procesando {uploaded_file.name}..."):
                    loaded_model = self.load_model_from_upload(uploaded_file)
                    if loaded_model:
                        st.experimental_rerun()
        
        st.markdown("**Opción 2: Ruta del Modelo (Recomendado para archivos grandes)**")

        model_path = st.text_input(
            "Ruta completa al modelo",
            value=str(self.config.DEFAULT_MODEL_PATH),
            help="Ejemplo: C:/ruta/completa/modelo.h5 o /home/user/modelo.h5",
            key="model_path_input"
        )

        path_exists = os.path.exists(model_path) if model_path else False
        if model_path and path_exists:
            try:
                file_size = os.path.getsize(model_path)
                st.success(f"Archivo encontrado ({file_size / (1024*1024):.1f} MB)")
            except:
                st.warning("No se puede obtener información del archivo")
        elif model_path:
            st.error("Archivo no encontrado en la ruta especificada")

        if st.button("Cargar desde Ruta", disabled=not path_exists, key="btn_path"):
            if path_exists:
                with st.spinner("Cargando modelo desde ruta..."):
                    loaded_model = self.load_model_from_path(model_path)
                    if loaded_model:
                        st.experimental_rerun()
            else:
                st.error("Especifica una ruta válida")

        st.markdown("**Opción 3: Descargar desde Link (Archivos Partidos)**")

        with st.expander("Instrucciones para archivos partidos"):
            st.markdown("""
            **Opción A - Desde repositorio de GitHub (Recomendado):**
            1. Copia el link del directorio de GitHub donde están las partes
            2. Pega el link completo (ej: https://github.com/user/repo/tree/main/split_model)
            3. El sistema detectará automáticamente todos los archivos .h5 y .part del directorio
            4. Se descargarán, unirán y cargarán automáticamente

            **Ejemplo:**
            ```
            https://github.com/user/clasificador-cancer-de-mama-RN/tree/main/split_model
            ```

            **Opción B - URLs directas:**
            1. Ingresa las URLs de todas las partes del modelo
            2. Puedes ingresar múltiples URLs separadas por comas o en líneas diferentes
            3. Las partes se descargarán y unirán automáticamente

            **Ejemplo con URLs separadas por comas:**
            ```
            https://ejemplo.com/modelo.h5.part1,
            https://ejemplo.com/modelo.h5.part2,
            https://ejemplo.com/modelo.h5.part3
            ```

            **Ejemplo con URLs en líneas separadas:**
            ```
            https://ejemplo.com/modelo.h5.part1
            https://ejemplo.com/modelo.h5.part2
            https://ejemplo.com/modelo.h5.part3
            ```
            """)

        model_urls = st.text_area(
            "URL del directorio de GitHub o URLs directas de las partes",
            height=100,
            placeholder="Ej: https://github.com/user/repo/tree/main/split_model\nO URLs directas separadas por comas o saltos de línea",
            help="Ingresa el link del repositorio de GitHub o las URLs directas de las partes del modelo",
            key="model_urls_input"
        )

        if model_urls:
            urls = self.file_downloader.extract_part_urls(model_urls)
            # st.info(f"Se detectaron {len(urls)} parte(s) para descargar")

            # for i, url in enumerate(urls, 1):
            #     st.text(f"Parte {i}: {url[:60]}..." if len(url) > 60 else f"Parte {i}: {url}")

        if st.button("Descargar y Unir Modelo", disabled=not bool(model_urls), key="btn_download"):
            if model_urls:
                urls = self.file_downloader.extract_part_urls(model_urls)

                if len(urls) == 0:
                    st.error("No se detectaron URLs válidas")
                else:
                    with st.spinner(f"Descargando y cargando {len(urls)} parte(s)..."):
                        # Usar el nuevo método que carga directamente en memoria
                        loaded_model = self.file_downloader.download_and_merge_in_memory(urls)

                        if loaded_model:
                            # Guardar el modelo en session_state
                            st.session_state.model_loaded = loaded_model
                            st.session_state.model_info = {
                                'source': 'download',
                                'original_name': 'modelo_descargado.h5',
                                'path': 'Cargado desde memoria',
                                'size_mb': 0,  # No tenemos el tamaño exacto
                                'input_shape': loaded_model.input_shape,
                                'output_shape': loaded_model.output_shape,
                                'total_params': loaded_model.count_params()
                            }
                            st.success("✓ Modelo descargado y cargado exitosamente")
                            st.experimental_rerun()
                        else:
                            st.error("Error descargando o cargando el modelo")
            else:
                st.error("Ingresa las URLs de las partes del modelo")

        st.markdown("**Opción 4: Limpiar y Resetear**")
        if st.button("Limpiar Cache y Resetear", key="btn_reset"):
            self.clear_model()
            st.success("Cache limpiado. Puedes cargar un nuevo modelo.")
            st.experimental_rerun()
        
        if current_model is not None:
            st.divider()
            self.show_model_info()
        
        return current_model
    
    def show_model_info(self):
        """Mostrar información detallada del modelo cargado"""
        st.markdown("**Modelo Cargado Exitosamente**")
        
        if st.session_state.model_info is None:
            st.info("Información del modelo no disponible")
            return
        
        try:
            info = st.session_state.model_info
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                input_shape = info['input_shape']
                st.metric("Entrada", f"{input_shape[1]}×{input_shape[2]}×{input_shape[3]}")
            
            with col2:
                output_shape = info['output_shape']
                st.metric("Salida", f"{output_shape[-1]} clases")
            
            with col3:
                total_params = info['total_params']
                st.metric("Parámetros", f"{total_params:,}")
            
            # Información adicional
            with st.expander("Información Detallada del Modelo"):
                st.markdown(f"""
                **Información del Modelo:**
                - **Origen:** {'Archivo subido' if info['source'] == 'upload' else 'Ruta local'}
                - **Archivo:** {info.get('original_name', 'N/A')}
                - **Tamaño:** {info['size_mb']:.1f} MB
                - **Ruta actual:** {info['path']}
                - **Forma de entrada:** {info['input_shape']}
                - **Forma de salida:** {info['output_shape']}
                - **Parámetros totales:** {info['total_params']:,}
                - **Clases disponibles:** {list(self.config.CLASS_MAPPING.values())}
                
                **Estado:**
                - Modelo listo para predicciones
                - Compatible con la aplicación
                - Persistente durante la sesión
                """)
                
        except Exception as e:
            st.warning(f"No se pudo obtener información detallada: {str(e)}")
            st.success("Modelo cargado y funcional")