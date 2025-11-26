import streamlit as st
import requests
import os
import tempfile
from pathlib import Path
from typing import List, Optional
import re
import json
import subprocess
import sys
import h5py
from io import BytesIO
import tensorflow as tf


class FileDownloader:
    """Clase para descargar y unir archivos .h5 partidos desde links"""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "model_parts"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def is_github_repo_url(self, url: str) -> bool:
        """Detecta si la URL es de un directorio de GitHub"""
        github_patterns = [
            r'github\.com/.+/.+/tree/.+',
            r'github\.com/.+/.+/blob/.+'
        ]
        return any(re.search(pattern, url) for pattern in github_patterns)

    def extract_github_files(self, github_url: str) -> List[dict]:
        """
        Extrae los archivos de un directorio de GitHub usando la API de GitHub.
        Retorna una lista de diccionarios con 'name' y 'download_url'.
        """
        try:
            # Convertir URL de GitHub a API URL
            # Ejemplo: https://github.com/user/repo/tree/branch/path
            # A: https://api.github.com/repos/user/repo/contents/path?ref=branch

            match = re.search(r'github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)', github_url)
            if not match:
                match = re.search(r'github\.com/([^/]+)/([^/]+)/tree/([^/]+)/?$', github_url)
                if match:
                    owner, repo, branch = match.groups()
                    path = ""
                else:
                    st.error("Formato de URL de GitHub no válido")
                    return []
            else:
                owner, repo, branch, path = match.groups()

            # Construir URL de la API
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            params = {'ref': branch}

            # st.info(f"Consultando repositorio: {owner}/{repo} (rama: {branch})")

            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()

            files_data = response.json()

            # Filtrar solo archivos (no directorios) que sean partes del modelo
            model_files = []
            for file_info in files_data:
                if file_info['type'] == 'file':
                    name = file_info['name']
                    # Buscar archivos que sean partes (.part, .partX, etc.) o archivos .h5
                    if re.search(r'\.(part\d*|h5)$', name, re.IGNORECASE):
                        model_files.append({
                            'name': name,
                            'download_url': file_info['download_url'],
                            'size': file_info.get('size', 0)
                        })

            # Ordenar por nombre para mantener el orden de las partes
            model_files.sort(key=lambda x: x['name'])

            return model_files

        except requests.exceptions.RequestException as e:
            st.error(f"Error consultando GitHub API: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Error procesando URL de GitHub: {str(e)}")
            return []

    def extract_part_urls(self, base_url: str) -> List[str]:
        """
        Extrae las URLs de las partes del archivo.
        Soporta:
        - URLs de directorios de GitHub
        - Lista de URLs separadas por comas o saltos de línea
        - URL única
        """
        urls = []

        base_url = base_url.strip()

        # Verificar si es una URL de GitHub
        if self.is_github_repo_url(base_url):
            # st.info("Detectado repositorio de GitHub, extrayendo archivos...")
            files = self.extract_github_files(base_url)

            if files:
                # st.success(f"Encontrados {len(files)} archivos en el repositorio")
                # for file_info in files:
                #     size_mb = file_info['size'] / (1024*1024)
                #     st.text(f"  - {file_info['name']} ({size_mb:.2f} MB)")

                urls = [file_info['download_url'] for file_info in files]
            else:
                st.warning("No se encontraron archivos de modelo en el repositorio")

            return urls

        # Si contiene comas o saltos de línea, asume que son múltiples URLs
        if ',' in base_url or '\n' in base_url:
            urls = [url.strip() for url in re.split(r'[,\n]', base_url) if url.strip()]
        else:
            # URL única
            urls = [base_url]

        return urls

    def download_file(self, url: str, destination: Path, progress_callback=None) -> bool:
        """
        Descarga un archivo desde una URL con barra de progreso.
        """
        try:
            # st.text(f"Conectando a: {url[:80]}...")

            # Configurar headers para mejor compatibilidad
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8KB chunks
            downloaded = 0

            # st.text(f"Descargando {total_size / (1024*1024):.2f} MB...")

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress_callback(downloaded / total_size)

            # Verificar que se descargó completamente
            actual_size = destination.stat().st_size
            if total_size > 0 and actual_size != total_size:
                st.error(f"Descarga incompleta: {actual_size}/{total_size} bytes")
                return False

            # st.success(f"✓ Descargado: {actual_size / (1024*1024):.2f} MB")
            return True

        except requests.exceptions.RequestException as e:
            st.error(f"Error de red descargando {url[:50]}: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Error inesperado descargando: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

    def merge_parts_system_command(self, part_files: List[Path], output_path: Path) -> bool:
        """
        Une archivos usando comandos del sistema (más confiable para binarios).
        """
        try:
            sorted_parts = sorted(part_files, key=lambda x: x.name)
            st.info(f"Uniendo {len(sorted_parts)} partes usando comando del sistema...")

            if sys.platform == 'win32':
                # Windows: usar copy /b
                parts_str = '+'.join([f'"{str(p)}"' for p in sorted_parts])
                cmd = f'copy /b {parts_str} "{str(output_path)}"'

                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode != 0:
                    st.error(f"Error ejecutando copy: {result.stderr}")
                    return False
            else:
                # Unix/Linux/Mac: usar cat
                with open(output_path, 'wb') as outfile:
                    for part_file in sorted_parts:
                        with open(part_file, 'rb') as infile:
                            outfile.write(infile.read())

            # Verificar el resultado
            if not output_path.exists():
                st.error("El archivo de salida no se creó")
                return False

            final_size = output_path.stat().st_size
            expected_size = sum(p.stat().st_size for p in sorted_parts)

            if final_size != expected_size:
                st.error(f"Error de integridad: {final_size} != {expected_size}")
                return False

            st.success(f"✓ Unión completada: {final_size / (1024*1024):.2f} MB")
            return True

        except Exception as e:
            st.error(f"Error en merge_parts_system_command: {str(e)}")
            return False

    def merge_parts(self, part_files: List[Path], output_path: Path) -> bool:
        """
        Une múltiples archivos partidos en uno solo.
        Intenta primero con Python puro, luego con comandos del sistema.
        """
        try:
            # Ordenar archivos para asegurar el orden correcto
            sorted_parts = sorted(part_files, key=lambda x: x.name)

            st.info(f"Uniendo {len(sorted_parts)} partes...")

            # Método 1: Python puro con escritura directa
            with open(output_path, 'wb') as output_file:
                for i, part_file in enumerate(sorted_parts):
                    if not part_file.exists():
                        st.error(f"El archivo {part_file.name} no existe")
                        return False

                    file_size = part_file.stat().st_size
                    if file_size == 0:
                        st.error(f"El archivo {part_file.name} está vacío")
                        return False

                    # Leer todo el archivo de una vez para evitar problemas de buffer
                    with open(part_file, 'rb') as input_file:
                        data = input_file.read()
                        output_file.write(data)

                    # st.text(f"✓ Parte {i+1}/{len(sorted_parts)} agregada")

            # Verificar el tamaño final
            final_size = output_path.stat().st_size
            expected_size = sum(p.stat().st_size for p in sorted_parts)

            if final_size != expected_size:
                st.warning(f"Integridad no coincide ({final_size} vs {expected_size}), intentando método alternativo...")
                output_path.unlink()
                return self.merge_parts_system_command(part_files, output_path)

            st.success(f"✓ Archivo unido exitosamente. Tamaño total: {final_size / (1024*1024):.2f} MB")

            # Validar que es un archivo HDF5 válido
            try:
                with h5py.File(str(output_path), 'r') as hdf5_file:
                    keys = list(hdf5_file.keys())
                    st.success(f"✓ Validación HDF5 exitosa - {len(keys)} grupos encontrados")
                return True
            except Exception as e:
                st.error(f"⚠️ El archivo unido no es un HDF5 válido: {str(e)}")
                st.warning("Esto puede indicar que las partes en GitHub están corruptas o mal partidas")
                return False

        except Exception as e:
            st.error(f"Error uniendo archivos: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

    def download_and_merge_in_memory(self, urls: List[str]) -> Optional[object]:
        """
        Descarga múltiples partes de un archivo y las une directamente en memoria.
        Retorna el modelo cargado de TensorFlow/Keras.

        Args:
            urls: Lista de URLs de las partes del archivo

        Returns:
            Modelo de TensorFlow/Keras o None si hay error
        """
        try:
            st.info(f"Iniciando descarga de {len(urls)} parte(s) en memoria...")

            model_bytes = b''

            # Descargar y combinar todas las partes directamente en memoria
            for i, url in enumerate(urls):
                # st.write(f"**📥 Descargando parte {i+1}/{len(urls)}**")

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }

                    response = requests.get(url, stream=True, timeout=60, headers=headers)
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    part_bytes = b''

                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            part_bytes += chunk
                            downloaded += len(chunk)

                            if total_size > 0:
                                progress = downloaded / total_size
                                progress_bar.progress(min(progress, 1.0))
                                status_text.text(f"Progreso: {progress*100:.1f}%")

                    # Agregar esta parte al modelo completo
                    model_bytes += part_bytes

                    progress_bar.empty()
                    status_text.empty()

                    # st.success(f"✓ Parte {i+1} descargada: {len(part_bytes) / (1024*1024):.2f} MB")

                except Exception as e:
                    st.error(f"Error descargando parte {i+1}: {str(e)}")
                    return None

            # Verificar que tenemos datos
            total_size_mb = len(model_bytes) / (1024*1024)
            st.success(f"✓ Total descargado: {total_size_mb:.2f} MB")

            # Crear archivo HDF5 en memoria y cargar el modelo
            st.info("🔗 Cargando modelo desde memoria...")

            try:
                # Método 1: Usar h5py para abrir el archivo HDF5 en memoria
                st.info("Método 1: Abriendo con h5py...")
                with h5py.File(BytesIO(model_bytes), 'r') as hf:
                    # Cargar el modelo desde el archivo HDF5
                    model = tf.keras.models.load_model(hf)

                st.success("✓ Modelo cargado exitosamente desde memoria (h5py)")
                return model

            except Exception as e:
                st.warning(f"Método h5py falló: {str(e)}")

                # Método 2: Intentar cargar directamente desde BytesIO
                st.info("Método 2: Intentando BytesIO directo...")
                try:
                    model_file = BytesIO(model_bytes)
                    model = tf.keras.models.load_model(model_file)

                    st.success("✓ Modelo cargado exitosamente desde memoria (BytesIO)")
                    return model

                except Exception as e2:
                    st.warning(f"Método BytesIO directo falló: {str(e2)}")

                    # Método 3: Guardar temporalmente y cargar
                    st.info("Método 3: Guardando temporalmente...")
                    temp_path = self.temp_dir / "temp_model.h5"

                    try:
                        with open(temp_path, 'wb') as f:
                            f.write(model_bytes)

                        st.info("Archivo temporal creado, cargando modelo...")
                        model = tf.keras.models.load_model(str(temp_path))

                        # Limpiar archivo temporal
                        if temp_path.exists():
                            temp_path.unlink()

                        st.success("✓ Modelo cargado exitosamente (archivo temporal)")
                        return model

                    except Exception as e3:
                        st.error(f"Todos los métodos fallaron. Último error: {str(e3)}")
                        if temp_path.exists():
                            temp_path.unlink()
                        return None

        except Exception as e:
            st.error(f"Error en el proceso de descarga y carga: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def download_and_merge(self, urls: List[str], output_filename: str = "merged_model.h5") -> Optional[str]:
        """
        Descarga múltiples partes de un archivo y las une.

        Args:
            urls: Lista de URLs de las partes del archivo
            output_filename: Nombre del archivo de salida

        Returns:
            Ruta del archivo unido o None si hay error
        """
        part_files = []
        output_path = self.temp_dir / output_filename

        try:
            st.info(f"Iniciando descarga de {len(urls)} parte(s)...")

            # Descargar cada parte
            for i, url in enumerate(urls):
                # st.write(f"**📥 Parte {i+1}/{len(urls)}**")

                part_filename = self.temp_dir / f"part_{i+1:03d}.tmp"

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress):
                    progress_bar.progress(min(progress, 1.0))
                    # status_text.text(f"Progreso: {progress*100:.1f}%")

                success = self.download_file(url, part_filename, update_progress)

                progress_bar.empty()
                status_text.empty()

                if not success:
                    st.error(f"Fallo al descargar parte {i+1}")
                    self.cleanup_temp_files(part_files)
                    return None

                part_files.append(part_filename)

            # Verificar que todas las partes se descargaron
            st.info(f"✓ Todas las partes descargadas. Total: {len(part_files)} archivo(s)")

            # Eliminar archivo de salida anterior si existe
            if output_path.exists():
                st.warning("Archivo anterior detectado, eliminando...")
                output_path.unlink()

            # Unir las partes
            st.info("🔗 Iniciando proceso de unión...")

            if self.merge_parts(part_files, output_path):
                # Verificar que el archivo existe y tiene contenido
                if not output_path.exists():
                    st.error("El archivo unido no se creó correctamente")
                    self.cleanup_temp_files(part_files)
                    return None

                total_size = output_path.stat().st_size

                if total_size == 0:
                    st.error("El archivo unido está vacío")
                    self.cleanup_temp_files(part_files)
                    if output_path.exists():
                        output_path.unlink()
                    return None

                st.success(f"✓ Modelo unido exitosamente: {total_size / (1024*1024):.2f} MB")

                # Limpiar archivos temporales de partes
                st.info("🧹 Limpiando archivos temporales...")
                self.cleanup_temp_files(part_files)

                return str(output_path)
            else:
                st.error("Error durante el proceso de unión")
                self.cleanup_temp_files(part_files)
                if output_path.exists():
                    output_path.unlink()
                return None

        except Exception as e:
            st.error(f"Error en el proceso de descarga y unión: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

            # Limpiar en caso de error
            self.cleanup_temp_files(part_files)
            if output_path.exists():
                try:
                    output_path.unlink()
                except:
                    pass

            return None

    def cleanup_temp_files(self, files: List[Path]):
        """Elimina archivos temporales."""
        for file_path in files:
            try:
                if file_path.exists():
                    os.unlink(file_path)
            except Exception as e:
                st.warning(f"No se pudo eliminar {file_path}: {str(e)}")

    def validate_url(self, url: str) -> bool:
        """Valida que una URL sea accesible."""
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except:
            return False
