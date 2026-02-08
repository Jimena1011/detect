# YOLOv8 + DeepSORT – Detección, Conteo y Estimación de Velocidad

Este es un proyecto de visión por computadora para la **detección, seguimiento, conteo y análisis de flujo
vehicular y peatonal**, utilizando **YOLOv8** y **DeepSORT**, desarrollado como parte del
programa de Asistencia Académica del Instituto Tecnológico de Costa Rica (TEC).

**Estado del proyecto:** en etapa de *prueba y error*.  
El sistema es funcional para detección, seguimiento y conteo; sin embargo, el cálculo de
**velocidades reales** continúa en fase experimental.

---

## Objetivo del proyecto

Desarrollar un sistema capaz de:

- Detectar personas y vehículos en video.
- Mantener identidades persistentes mediante seguimiento (tracking).
- Contar objetos según su dirección de movimiento.
- Explorar métodos para estimar velocidad real a partir de video.
- Evaluar el rendimiento del sistema en distintos escenarios urbanos.

---

## Tecnologías utilizadas

- **YOLOv8** – detección de objetos
- **DeepSORT** – seguimiento multiobjeto
- **OpenCV** – procesamiento de video
- **PyTorch** – inferencia del modelo
- **NumPy** – operaciones matemáticas

---
## Versiones principales del sistema

El desarrollo del proyecto se realizó mediante **versiones incrementales**, con el fin de
no perder avances y poder analizar los resultados de cada enfoque.

## Limitaciones actuales

- La **estimación de velocidad real** depende fuertemente del ángulo de la cámara.
- La homografía requiere recalibración para cada escenario.
- Cambios pequeños en la grabación afectan los resultados.
- No existe aún un método universal y confiable para todos los videos.

---

## Ejecución básica

Ejemplo general:

```bash
python predict1.py model=yolov8l.pt source="pavas_1.MOV" show=True