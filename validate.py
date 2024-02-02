import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Cargar el modelo entrenado
model = load_model('model_dogs_cats.h5')

# Directorio de datos de prueba
test_dir = 'data/test'

# Obtener la lista de archivos en el directorio de prueba
test_files = os.listdir(test_dir)

# Realizar la predicción para cada imagen en el directorio de prueba
for filename in test_files:
    img_path = os.path.join(test_dir, filename)
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión adicional para el lote
    img_array /= 255.0  # Normalizar los valores de píxeles
    prediction = model.predict(img_array)
    
    # Imprimir el resultado de la predicción
    if prediction[0][0] > 0.5:
        print(f"{filename} es un perro")
    else:
        print(f"{filename} es un gato")
