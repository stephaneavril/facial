# Facial Mask Detector - Demo (Flask + OpenCV)

Este repo contiene una aplicación mínima en Flask que compara una foto subida con una imagen objetivo (máscara).
Si la comparación encuentra suficientes coincidencias, la aplicación muestra el código `4789#`.

## Archivos

- `app.py` - servidor Flask.
- `templates/index.html` - página principal.
- `static/target_mask.png` - imagen objetivo (la máscara que se detecta).
- `requirements.txt` - dependencias.
- `Procfile` - para desplegar en Render (o Heroku-like).
- `README.md` - este archivo.

## Uso local

1. Crear virtualenv e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ejecutar:
   ```bash
   python app.py
   ```

3. Abrir `http://127.0.0.1:5000` y subir una foto.

## Despliegue en Render

- Crear un nuevo servicio web y conectar a tu repositorio GitHub.
- El comando de build puede ser:
  ```
  pip install -r requirements.txt
  ```
- El comando de start es `gunicorn app:app`.

## Advertencias importantes

- **Privacidad & seguridad**: Este proyecto es un prototipo. Usar reconocimiento facial para control de acceso requiere cumplimiento legal y medidas de seguridad fuertes.
- **Robustez**: ORB matching es simple y no confiable en producción (variaciones de luz, ángulo, calidad).
