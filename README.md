# Evaluación Unidad 2 — Extracción de Conocimiento de Base de Datos

Evaluación de la Unidad 2 de la materia **Extracción de Conocimiento de Base de Datos** (UTEQ · Ingeniería en Gestión de Desarrollo de Software · 2026).

**Docente:** Filiberto Ruíz Hernández
**Alumno:** Mario Alberto Ramírez Martínez — Matrícula 2023371044

## Contenido

| # | Ejercicio | Dataset | Algoritmo | Notebook |
|---|-----------|---------|-----------|----------|
| 1 | SA · Regresión | `beisbol.csv` | Regresión Lineal Simple | `notebooks/1_sa_regresion_lineal.ipynb` |
| 2 | SA · Clasificación | `breast-cancer.csv` | k-NN | `notebooks/2_sa_clasificacion_knn.ipynb` |
| 3 | NS · Agrupación | `samsung.csv` | K-Means | `notebooks/3_ns_agrupacion_kmeans.ipynb` |
| 4 | NS · Reducción de dimensionalidad | `comprar_alquilar.csv` | PCA | `notebooks/4_ns_pca.ipynb` |
| 5 | DE · Clasificación (alterno) | `breast-cancer.csv` | Random Forest | `notebooks/5_de_clasificacion_randomforest.ipynb` |
| 6 | DE · Regresión (alterno) | `beisbol.csv` | Random Forest | `notebooks/6_de_regresion_randomforest.ipynb` |
| 7 | AU · Herramienta alternativa | SA y DE | App Streamlit | `app/streamlit_app.py` |

## Estructura

```
evaluacion_u2/
├── data/raw/                 # Conjuntos de datos de la evaluación
├── notebooks/                # Notebooks ejecutados (SA, NS, DE)
├── modelos/                  # Modelos entrenados (*.pkl)
├── app/                      # Herramienta alternativa (Streamlit)
├── reporte/                  # Reporte LaTeX + PDF + figuras
└── requirements.txt
```

## Reproducción

```bash
uv venv .venv
uv pip install -r requirements.txt
# App AU (herramienta alternativa):
streamlit run app/streamlit_app.py
# Ejecutar notebooks:
jupyter nbconvert --execute --inplace notebooks/*.ipynb
```

## Reporte

El reporte completo está en `reporte/reporte-evaluacion-u2.pdf` (fuente `.tex`).

## Licencia / Nota

Material académico de la evaluación de la Unidad 2. Datasets proporcionados por el docente.
