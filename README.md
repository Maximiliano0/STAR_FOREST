# STAR_FOREST

Proyecto educativo de clasificación de estrellas para un curso de Machine
Learning. Implementa un pipeline completo de EDA, entrenamiento de un
`DecisionTreeClassifier` y una GUI interactiva (Tkinter) para predicción.

## Contenido del proyecto

**Módulos principales (en `scripts/`):**

- `scripts/eda_pipeline.py`: Preparación de datos (descarga, normalización
  de texto, validación por dominio, eliminación de outliers por IQR,
  análisis de desbalance y oversampling condicional).
- `scripts/model_train.py`: Entrenamiento del `DecisionTreeClassifier` y
  guardado de artefactos. Entrena y testea sobre el dataset
  **limpio + balanceado**, con split estratificado.
- `scripts/star_gui.py`: Aplicación interactiva con interfaz gráfica.
- `scripts/config.py`: Constantes compartidas (features, tipos de estrella,
  hiperparámetros del modelo).

**Datos y resultados:**

- `data/`: Datasets (raw, limpio, balanceado).
- `models/`: Artefactos entrenados (modelo + encoders + columnas).
- `reports/`: Gráficos del EDA y de evaluación del modelo.
- `pictures/`: Imágenes de los tipos de estrellas (usadas por la GUI).
- `utils/`: Configuración de tooling (`.pylintrc`, `pyrightconfig.json`,
  `requirements.txt`).

## Objetivo didáctico

Pipeline básico de Machine Learning explicado paso a paso:

1. Carga de datos desde un CSV remoto (cacheado localmente).
2. Limpieza categórica (normalización de `Star color`).
3. Validación por dominio (rangos físicos plausibles).
4. Eliminación de outliers por IQR (por clase).
5. Oversampling **condicional** (solo si el dataset queda desbalanceado).
6. Entrenamiento de `DecisionTreeClassifier` y evaluación
   (accuracy, matriz de confusión, precision/recall/F1).
7. Predicción interactiva en una GUI Tkinter.

## Requisitos

- Python 3.9+ (probado con 3.12).
- Paquetes en `utils/requirements.txt`.

```bash
pip install -r utils/requirements.txt
```

## Pipeline de ejecución

Los tres pasos son independientes. Todos asumen que se ejecutan desde la
raíz del proyecto.

### Paso 1 — EDA Pipeline

```bash
python scripts/eda_pipeline.py
```

Realiza:

- Descarga local de `data/Stars.csv` (si no existe).
- Normalización de `Star color` (p. ej. `yellow-white` → `Yellow`,
  `Blue-white` → `Blue White`).
- **Validación por dominio:** descarta filas con NaN o valores fuera de
  rangos físicos plausibles
  (Temperature ∈ [1.000, 60.000] K, Luminosity ∈ [1e-5, 1e7] L⊙,
  Radius ∈ [0,01, 2.500] R⊙, Magnitud absoluta ∈ [-15, 25],
  Star type ∈ {0..5}).
- **Eliminación de outliers por IQR** (k = 1,5) calculada
  **por clase**, para preservar la variabilidad legítima entre tipos
  estelares.
- **Oversampling condicional:** solo si tras la limpieza el dataset
  sigue desbalanceado (`max_count > min_count`).

**Salida:**

- `data/Stars_clean.csv` — dataset limpio (sin oversampling).
- `data/Stars_balanced.csv` — dataset listo para entrenar
  (limpio + oversampling si corresponde).
- Gráficos en `reports/` (ver más abajo).

### Paso 2 — Entrenamiento del modelo

```bash
python scripts/model_train.py
```

Realiza:

- Carga el dataset limpio y balanceado mediante
  `prepare_dataset_for_model(use_balanced=True)`.
- Codifica `Star color` y `Spectral Class` con `LabelEncoder`.
- Split **estratificado** 80/20 sobre el dataset balanceado.
- Entrena `DecisionTreeClassifier` (`max_depth=5`, `ccp_alpha=0.01`,
  `min_samples_split=10`, `random_state=42`).
- Reporta accuracy, matriz de confusión y `classification_report`.
- Genera tres visualizaciones de evaluación.

**Salida:**

- `models/star_classifier.pkl`
- `models/le_color.pkl`
- `models/le_spectral.pkl`
- `models/X_features.pkl`
- `reports/model_confusion_matrix.png`
- `reports/model_metrics_by_class.png`
- `reports/model_accuracy_summary.png`

### Paso 3 — GUI

```bash
python scripts/star_gui.py
```

- Carga el modelo pre-entrenado (lo entrena si no existe).
- Permite ingresar parámetros de una estrella y predecir el tipo.
- Muestra una imagen del tipo predicho desde `pictures/`.
- Botón "View Tree Logic" para visualizar el árbol de decisión.

## Ejecución rápida

Primera vez:

```bash
python scripts/eda_pipeline.py
python scripts/model_train.py
python scripts/star_gui.py
```

Posteriores (modelo ya entrenado):

```bash
python scripts/star_gui.py
```

## Estructura del proyecto

```text
STAR_FOREST/
├── README.md
├── .gitignore
│
├── scripts/
│   ├── config.py              (Constantes compartidas)
│   ├── eda_pipeline.py        (Paso 1: EDA + limpieza)
│   ├── model_train.py         (Paso 2: Entrenamiento)
│   └── star_gui.py            (Paso 3: GUI)
│
├── data/
│   ├── Stars.csv              (Dataset raw)
│   ├── Stars_clean.csv        (Limpio: wording + dominio + IQR)
│   └── Stars_balanced.csv     (Limpio + oversampling condicional)
│
├── models/
│   ├── star_classifier.pkl
│   ├── le_color.pkl
│   ├── le_spectral.pkl
│   └── X_features.pkl
│
├── reports/
│   ├── wording_fixes_star_color.png
│   ├── domain_invalid_removed.png
│   ├── outliers_boxplots_before_after.png
│   ├── feature_distributions.png
│   ├── class_imbalance_before_after.png
│   ├── model_confusion_matrix.png
│   ├── model_metrics_by_class.png
│   └── model_accuracy_summary.png
│
├── pictures/
│   ├── brown_dwarf.png
│   ├── red_dwarf.png
│   ├── white_dwarf.png
│   ├── main_sequence.png
│   ├── supergiant.png
│   └── hypergiant.png
│
└── utils/
    ├── requirements.txt
    ├── .pylintrc
    └── pyrightconfig.json
```

## Visualizaciones (propósito educativo)

El proyecto genera reportes pensados para que estudiantes vean cada etapa
del pipeline.

**EDA (`scripts/eda_pipeline.py`):**

- `wording_fixes_star_color.png` — Variantes de texto detectadas y su
  normalización (`Blue-white` → `Blue White`, `yellow-white` → `Yellow`,
  etc.).
- `domain_invalid_removed.png` — Filas eliminadas por cada regla de
  dominio (rangos físicos inválidos).
- `outliers_boxplots_before_after.png` — Boxplots por feature antes y
  después del filtro IQR (escala logarítmica para Temperature, Luminosity
  y Radius).
- `feature_distributions.png` — Histogramas de las 4 features numéricas
  agrupados por `Star type`.
- `class_imbalance_before_after.png` — Distribución de clases antes y
  después del oversampling condicional.

**Evaluación del modelo (`scripts/model_train.py`):**

- `model_confusion_matrix.png` — Heatmap de la matriz de confusión.
- `model_metrics_by_class.png` — Precision, Recall y F1 por clase.
- `model_accuracy_summary.png` — Resumen de accuracy global +
  recall por clase.

## Resumen del pipeline (estado actual)

Comportamiento sobre el dataset original (240 filas):

1. **Wording fixes** en `Star color`: ~52 reemplazos.
2. **Validación por dominio:** conserva 221/240 filas (19 eliminadas).
3. **IQR (k = 1,5) por clase:** conserva 197/221 filas (24 eliminadas).
4. **Oversampling condicional:** 6 clases × 37 muestras = 222 filas en
   `Stars_balanced.csv`.
5. **Split estratificado** 80/20 sobre el dataset balanceado
   (177 train / 45 test).

El modelo actual obtiene **100% de accuracy** en el conjunto de test
estratificado.

## Notas

- Dataset fuente:
  [YBIFoundation Dataset](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Stars.csv).
- Las imágenes en `pictures/` deben mantener su nombre
  (`<star_type>.png`) para que la GUI las localice.
- Tooling configurado en `utils/`: `pylint --rcfile=utils/.pylintrc` y
  `pyright --project utils/pyrightconfig.json` deben pasar sin errores.
