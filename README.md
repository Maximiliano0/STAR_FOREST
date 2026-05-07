# STAR_FOREST

Proyecto educativo de clasificacion de estrellas para un curso de Machine Learning.

## Contenido del proyecto

**Módulos principales:**

- `eda_pipeline.py`: Preparación de datos (descarga, limpieza, análisis de imbalance, oversampling)
- `model_train.py`: Entrenamiento del modelo y guardado de artefactos
- `star_gui.py`: Aplicación interactiva con interfaz gráfica (Tkinter)

**Datos y resultados:**

- `data/`: Datasets (raw, limpio, balanceado)
- `models/`: Artefactos entrenados (modelo, encoders)
- `reports/`: Gráficos de análisis
- `pictures/`: Imágenes de tipos de estrellas para la GUI

## Objetivo didactico

Este proyecto esta pensado para explicar, de forma practica, un pipeline basico de Machine Learning:

1. Carga de datos desde un CSV.
2. Preparacion de variables numericas y categoricas.
3. Entrenamiento de un `DecisionTreeClassifier`.
4. Evaluacion con accuracy, matriz de confusion y reporte de clasificacion.
5. Uso del modelo entrenado dentro de una interfaz grafica interactiva.

## Requisitos

- Python 3.9+
- Paquetes de Python (ver `requirements.txt`)

Instalacion sugerida:

```bash
pip install -r requirements.txt
```

## Pipeline de Ejecucion

El proyecto sigue un pipeline de 3 pasos independientes:

### Paso 1: EDA Pipeline (Preparacion de Datos)

```bash
python eda_pipeline.py
```

Realiza:

- Descarga local del dataset
- Limpieza de problemas de wording
- Analisis de desbalance de clases
- Oversampling
- Genera graficos de analisis

**Salida:** `data/Stars_clean.csv`, `data/Stars_balanced.csv`, graficos en `reports/`

### Paso 2: Model Training

```bash
python model_train.py
```

Realiza:

- Carga del dataset limpio
- Entrenamiento del DecisionTreeClassifier
- Guardado de modelo y encoders

**Salida:** Modelo entrenado en `models/` (modelo, encoders, feature columns)

### Paso 3: GUI Application

```bash
python star_gui.py
```

- Carga el modelo pre-entrenado
- Interfaz interactiva para predicciones
- Visualizacion del arbol de decision

## Ejecucion Rapida (Todos los pasos)

Si es la primera ejecucion:

```bash
python eda_pipeline.py
python model_train.py
python star_gui.py
```

Para ejecuciones posteriores (el modelo ya esta entrenado):

```bash
python star_gui.py
```

## Cómo Usar los Scripts

### 1️⃣ EDA Pipeline - `eda_pipeline.py`

**Propósito:** Descargar, limpiar y analizar datos, generar reportes.

**Cuándo usar:**

- Primera vez que ejecutas el proyecto
- Cuando necesitas regenerar los datos limpios y balanceados
- Para analizar la calidad del dataset

**Uso:**

```bash
python eda_pipeline.py
```

**Qué hace:**

```text
✓ Descarga Stars.csv (si no existe localmente)
✓ Carga y limpia problemas de redacción (wording fixes)
✓ Detecta 48 inconsistencias en datos
✓ Analiza desbalance de clases
✓ Aplica oversampling para balancear
✓ Genera gráficos en reports/
```

**Archivos generados:**

- `data/Stars.csv` - Dataset original
- `data/Stars_clean.csv` - Dataset limpio
- `data/Stars_balanced.csv` - Dataset balanceado
- `reports/wording_fixes_star_color.png` - Gráfico de correcciones
- `reports/class_imbalance_before_after.png` - Análisis de balance

**Ejemplo de salida en terminal:**

```text
Descargando dataset desde URL...
Dataset descargado correctamente: 240 registros
Detectados 48 problemas de wording en Star color
Balanceando dataset con oversampling...
✓ Gráficos guardados en reports/
```

---

### 2️⃣ Model Training - `model_train.py`

**Propósito:** Entrenar el modelo DecisionTree, evaluar y generar visualizaciones de métricas.

**Cuándo usar:**

- Después de ejecutar `eda_pipeline.py`
- Cuando necesitas reentrear el modelo
- Para generar reportes de evaluación del modelo

**Uso:**

```bash
python model_train.py
```

**Qué hace:**

```text
✓ Carga datos limpios
✓ Entrena DecisionTreeClassifier
✓ Evalúa en test set (80/20 split)
✓ Calcula accuracy, confusion matrix, precision/recall/f1
✓ Genera 3 visualizaciones en reports/
✓ Guarda modelo y encoders en models/
```

**Métricas mostradas en terminal:**

```text
Accuracy Global: 100.00%

Confusion Matrix:
[[8  0  0  0  0  0]
 [0  7  0  0  0  0]
 ...

Precisión, Recall, F1-Score por clase
```

**Archivos generados:**

- `models/star_classifier.pkl` - Modelo entrenado
- `models/le_color.pkl` - Encoder de colores
- `models/le_spectral.pkl` - Encoder de clase espectral
- `models/X_features.pkl` - Nombres de features
- `reports/model_confusion_matrix.png` - Matriz de confusión (heatmap)
- `reports/model_metrics_by_class.png` - Métricas por clase (gráfico barras)
- `reports/model_accuracy_summary.png` - Resumen de accuracy (gauge)

**Ejemplo de salida:**

```text
MODEL TRAINING COMPLETED
Accuracy Global: 100.00%
Generando visualizaciones...
✅ Confusion matrix plot: .../reports/model_confusion_matrix.png
✅ Metrics by class plot: .../reports/model_metrics_by_class.png
✅ Accuracy summary plot: .../reports/model_accuracy_summary.png
Model saved to: .../models/star_classifier.pkl
```

---

### 3️⃣ GUI Application - `star_gui.py`

**Propósito:** Interfaz gráfica interactiva para hacer predicciones de tipos de estrellas.

**Cuándo usar:**

- Después de entrenar el modelo
- Para demostrar el modelo a usuarios
- Para hacer predicciones sobre nuevas estrellas

**Uso:**

```bash
python star_gui.py
```

**Interfaz:**

```text
┌─ STAR TYPE PREDICTOR ────────────────────┐
│                                          │
│ Temperature (K):        [input field]    │
│ Luminosity (L/Lo):      [input field]    │
│ Radius (R/Ro):          [input field]    │
│ Absolute Magnitude:     [input field]    │
│ Star Color:             [dropdown]       │
│ Spectral Class:         [dropdown]       │
│                                          │
│ [PREDICT STAR TYPE]  [View Tree Logic]   │
│                                          │
│ Prediction: Brown Dwarf                  │
│ [Imagen de la estrella]                  │
│                                          │
└──────────────────────────────────────────┘
```

**Cómo usar:**

1. Ingresa valores numéricos para los 6 primeros campos
2. Selecciona color de la estrella (ej: "White", "Red", "Yellow")
3. Selecciona clase espectral (ej: "M", "K", "A")
4. Haz clic en "PREDICT STAR TYPE"
5. Ver predicción e imagen del tipo de estrella

**Ejemplo de predicción:**

```text
Input: Temperature=5778, Luminosity=1.0, Radius=1.0,
       Magnitude=4.83, Color=Yellow, Spectral=G
Output: Main Sequence ⭐
```

---

## Flujo Típico de Uso

### Primera Vez (Configuración Inicial)

```bash
# Paso 1: Preparar datos
python eda_pipeline.py
# Espera: descarga, limpia, genera reportes

# Paso 2: Entrenar modelo
python model_train.py
# Espera: entrena y genera visualizaciones de métricas

# Paso 3: Usar la GUI
python star_gui.py
# Interactúa con la interfaz gráfica
```

### Ejecuciones Posteriores (Modelo Ya Entrenado)

```bash
# Solo abre la GUI (el modelo ya existe)
python star_gui.py
```

---

## Regenerar Reportes

Si necesitas regenerar los gráficos de análisis:

```bash
# Regenerar análisis de datos (EDA)
python eda_pipeline.py

# Regenerar análisis de modelo (métricas y visualizaciones)
python model_train.py
```

Los gráficos se guardarán automáticamente en `reports/` y sobrescribirán los anteriores.

---

## Estructura del Proyecto

```text
STAR_FOREST/
├── README.md                      (Este archivo)
├── requirements.txt               (Dependencias)
├── EDA_REPORT.md                  (Reporte de EDA)
│
├── eda_pipeline.py                (Paso 1: Preparacion de datos)
├── model_train.py                 (Paso 2: Entrenamiento del modelo)
├── star_gui.py                    (Paso 3: Aplicacion GUI)
│
├── data/                          (Datos)
│   ├── Stars.csv                  (Dataset raw)
│   ├── Stars_clean.csv            (Dataset limpio)
│   └── Stars_balanced.csv         (Dataset oversampled)
│
├── models/                        (Artefactos del modelo)
│   ├── star_classifier.pkl        (Modelo entrenado)
│   ├── le_color.pkl               (Encoder de colores)
│   ├── le_spectral.pkl            (Encoder de clase espectral)
│   └── X_features.pkl             (Nombres de features)
│
├── reports/                       (Reportes de analisis e imagenes)
│   ├── wording_fixes_star_color.png           (Grafico de wording fixes)
│   ├── class_imbalance_before_after.png       (Grafico de imbalance)
│   ├── model_confusion_matrix.png             (Matriz de confusion del modelo)
│   ├── model_metrics_by_class.png             (Precision/Recall/F1 por clase)
│   └── model_accuracy_summary.png             (Resumen de accuracy)
│
└── pictures/                      (Imagenes de estrellas)
    ├── brown_dwarf.png
    ├── red_dwarf.png
    ├── white_dwarf.png
    ├── main_sequence.png
    ├── supergiant.png
    └── hypergiant.png
```

## Visualizaciones y Métricas (Propósito Educativo)

Este proyecto genera visualizaciones detalladas para ayudar a estudiantes a comprender:

**EDA Visualizations (después de ejecutar `eda_pipeline.py`):**

- `wording_fixes_star_color.png` - Muestra los 48 problemas de redacción detectados y corregidos
- `class_imbalance_before_after.png` - Compara la distribución de clases antes y después de oversampling

**Model Evaluation Visualizations (después de ejecutar `model_train.py`):**

- `model_confusion_matrix.png` - Heatmap de la matriz de confusión (muestra cuáles tipos de estrellas se confunden entre sí)
- `model_metrics_by_class.png` - Gráfico de barras con Precision, Recall y F1-Score para cada clase
- `model_accuracy_summary.png` - Resumen general: accuracy global + recall por clase

Estas visualizaciones están diseñadas para que los estudiantes entiendan:

- Cómo se limpian los datos
- Cómo se maneja el desbalance de clases
- Cómo evaluar modelos de clasificación
- Qué tipos de estrellas el modelo clasifica correctamente

---

## EDA, Limpieza y Oversampling

Para detalles completos sobre el Exploratory Data Analysis, ver [EDA_REPORT.md](EDA_REPORT.md).

El pipeline incluye automaticamente:

1. Descarga local del dataset en `data/Stars.csv`
2. Deteccion y normalizacion de problemas de redaccion en `Star color`
3. Analisis de desbalance de clases
4. **Oversampling condicional:** Se aplica oversampling SOLO si el dataset limpio tiene clases desbalanceadas. Si el dataset ya está balanceado, no se realiza oversampling adicional.
5. Guardado de graficos del analisis

## Notas

- El dataset fuente se descarga desde:
  - [YBIFoundation Dataset](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Stars.csv)
- La GUI ahora usa el dataset local con limpieza de wording para entrenar el modelo.
- Las imagenes deben permanecer dentro de la carpeta `pictures/` para que la GUI las encuentre correctamente.
