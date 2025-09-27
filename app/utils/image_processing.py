import numpy as np
import tensorflow as tf
from PIL import Image
from app.config import Config

class ImageProcessor:
    def __init__(self):
        self.config = Config()
    
    def load_image(self, uploaded_file):
        """Cargar imagen desde archivo subido"""
        return Image.open(uploaded_file)
    
    def preprocess_image(self, image):
        """Preprocesar imagen para el modelo"""
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3:
            if img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            elif img_array.shape[-1] != 3:
                raise ValueError(f"Número de canales no soportado: {img_array.shape[-1]}")
        
        img_array = img_array.astype(np.float32) / 255.0
        img_array = tf.image.resize(img_array, self.config.INPUT_SIZE[:2])
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_image(self, image, model):
        """Predecir clase de una imagen"""
        try:
            processed_image = self.preprocess_image(image)
            
            predictions = model.predict(processed_image)
            probabilities = predictions[0]
            
            predicted_class_index = np.argmax(probabilities)
            predicted_class = self.config.CLASS_MAPPING[predicted_class_index]
            confidence = probabilities[predicted_class_index]
            
            return {
                'Prediccion': predicted_class,
                'Confianza': f"{confidence:.4f}",
                'Prob_Benign': f"{probabilities[0]:.4f}",
                'Prob_Malignant': f"{probabilities[1]:.4f}",
                'Prob_Normal': f"{probabilities[2]:.4f}"
            }
            
        except Exception as e:
            raise ValueError(f"Error en predicción: {str(e)}")