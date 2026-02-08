# YOLOv8 + DeepSORT – Detección, Conteo y Estimación de Velocidad

Proyecto de visión por computadora para la **detección, seguimiento, conteo y análisis de flujo
vehicular y peatonal**, utilizando **YOLOv8** y **DeepSORT**, desarrollado como parte del
programa de Asistencia Académica del Instituto Tecnológico de Costa Rica (TEC).

⚠️ **Estado del proyecto:** en etapa de *prueba y error*.  
El sistema es funcional para detección, seguimiento y conteo; sin embargo, el cálculo de
**velocidades reales** continúa en fase experimental.

---

## 📌 Objetivo del proyecto

Desarrollar un sistema capaz de:

- Detectar personas y vehículos en video.
- Mantener identidades persistentes mediante seguimiento (tracking).
- Contar objetos según su dirección de movimiento.
- Explorar métodos para estimar velocidad real a partir de video.
- Evaluar el rendimiento del sistema en distintos escenarios urbanos.

---

## 🧠 Tecnologías utilizadas

- **YOLOv8** – detección de objetos
- **DeepSORT** – seguimiento multiobjeto
- **OpenCV** – procesamiento de video
- **PyTorch** – inferencia del modelo
- **NumPy** – operaciones matemáticas
- **PostgreSQL** (opcional) – almacenamiento de datos
- **CUDA** (opcional) – aceleración por GPU

---
