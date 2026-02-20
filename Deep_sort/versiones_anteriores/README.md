# Versiones experimentales del sistema

Esta carpeta contiene **versiones intermedias y experimentales** del sistema de
detección, seguimiento, conteo y estimación de velocidad.
---

## Descripción de cada archivo

---

## 🔹 `predict_original.py`
- Versión base original del proyecto

**Funciones**
- Detección de objetos con YOLOv8
- Seguimiento multiobjeto con DeepSORT
- Visualización de bounding boxes
- Asignación de IDs por objeto
- Dibujo de trayectorias (trails)

**Características**
- Código cercano a ejemplos iniciales
- Sin conteo
- Sin estimación de velocidad
- Sin base de datos

---

## 🔹 `predict.py`
**Funciones**
- Detección y seguimiento (YOLOv8 + DeepSORT)
- Visualización en tiempo real
- Cálculo de FPS
- Manejo básico de interacción (tecla `q`)
- Preparación para extensiones futuras

**Características**
- Sin conteo formal
- Sin velocidad
- Usado como punto de partida para nuevas versiones

---

## 🔹 `predict_cero.py`
**Funciones**
- Detección y seguimiento
- Definición de puntos para homografía
- Conversión de coordenadas de píxeles a mundo real
- Exploración de perspectiva

**Características**
- No realiza conteo
- No calcula velocidad final
- Enfocado únicamente en la geometría del plano

---

## 🔹 `predict_count.py`
**Funciones**
- Detección y tracking
- Conteo por cruce de línea
- Conteo por clase (carros, buses, personas, motocicletas, bicicletas)
- Identificación de dirección de movimiento
- Visualización de líneas de conteo
- Visualización de contadores en pantalla

**Características**
- No estima velocidad
- Enfocado en flujo vehicular
- Optimización visual (menos overlays)

---

## 🔹 `predict_v_chat.py`
**Funciones**
- Detección y seguimiento
- Cálculo de desplazamiento entre frames
- Estimación de velocidad basada en FPS
- Visualización de velocidad estimada

**Características**
- Velocidades aproximadas
- Dependencia fuerte del FPS
- No usa homografía
- Uso exploratorio

---