"""Evaluación U2 — Herramienta alternativa a Jupyter Notebook.

App interactiva (Streamlit) que resuelve los ejercicios de SA y DE de la
evaluación de la Unidad 2 de Extracción de Conocimiento de Base de Datos,
utilizando una herramienta distinta a Jupyter Notebook.

Materia: Extracción de Conocimiento de Base de Datos
Docente: Filiberto Ruíz Hernández
Alumno: Mario Alberto Ramírez Martínez — Matrícula 2023371044
"""

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, silhouette_score,
    davies_bouldin_score,
)

matplotlib.use("Agg")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "raw")

st.set_page_config(page_title="Evaluación U2 — Extracción de Conocimiento", layout="wide")

st.title("Evaluación Unidad 2 — Extracción de Conocimiento de Base de Datos")
st.caption("Herramienta alternativa a Jupyter Notebook (Streamlit) · SA y DE")
st.write("**Materia:** Extracción de Conocimiento de Base de Datos — **Docente:** Filiberto Ruíz Hernández")
st.write("**Alumno:** Mario Alberto Ramírez Martínez (2023371044) — UTEQ · Ingeniería en Gestión de Desarrollo de Software")


@st.cache_data
def load_beisbol():
    return pd.read_csv(os.path.join(DATA, "beisbol.csv"), index_col=0)


@st.cache_data
def load_breast():
    return pd.read_csv(os.path.join(DATA, "breast-cancer.csv"))


@st.cache_data
def load_samsung():
    df = pd.read_csv(os.path.join(DATA, "samsung.csv"))
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_comprar():
    return pd.read_csv(os.path.join(DATA, "comprar_alquilar.csv"))


def metricas_regresion(y, yp):
    return {
        "R²": round(r2_score(y, yp), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y, yp)), 4),
        "MAE": round(mean_absolute_error(y, yp), 4),
    }


def metricas_clasificacion(y, yp):
    return {
        "Accuracy": round(accuracy_score(y, yp), 4),
        "Precision": round(precision_score(y, yp), 4),
        "Recall": round(recall_score(y, yp), 4),
        "F1": round(f1_score(y, yp), 4),
    }


def show_metricas(met):
    cols = st.columns(len(met))
    for col, (k, v) in zip(cols, met.items()):
        col.metric(k, v)


def plot_reales_vs_predichos(y, yp, titulo):
    fig, ax = plt.subplots()
    ax.scatter(y, yp, s=90, color="#2E86AB", label="Observaciones")
    lim = [min(y.min(), yp.min()) - 5, max(y.max(), yp.max()) + 5]
    ax.plot(lim, lim, "--", color="#A23B72", label="Línea ideal (y=x)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Valores reales"); ax.set_ylabel("Valores predichos")
    ax.set_title(titulo); ax.legend()
    st.pyplot(fig)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["1 · SA Regresión (Lineal)", "2 · SA Clasificación (k-NN)",
     "3 · NS Agrupación (K-Means)", "4 · NS Reducción (PCA)",
     "5 · DE Clasificación (Random Forest)", "6 · DE Regresión (Random Forest)"]
)

# ---------------- 1 · SA Regresión lineal ----------------
with tab1:
    st.header("1. SA — Regresión Lineal sobre beisbol.csv")
    st.markdown("Predicción de carreras (`runs`) a partir de los turnos al bat (`bateos`) con **Regresión Lineal Simple**.")
    df = load_beisbol()
    st.dataframe(df.head(10))
    X, y = df[["bateos"]], df["runs"]
    modelo = LinearRegression().fit(X, y)
    yp = modelo.predict(X)
    st.markdown(f"**Ecuación:** `runs = {modelo.coef_[0]:.4f} × bateos + {modelo.intercept_:.4f}`")
    show_metricas(metricas_regresion(y, yp))
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="bateos", y="runs", s=90, ax=ax)
    xline = np.linspace(X.min(), X.max(), 50)
    ax.plot(xline, modelo.predict(xline.reshape(-1, 1)), color="#A23B72", lw=2, label="Regresión lineal")
    ax.set_title("Comportamiento de los datos y recta ajustada"); ax.legend()
    st.pyplot(fig)
    plot_reales_vs_predichos(y, yp, "Valores reales vs predichos — Regresión Lineal")

# ---------------- 2 · SA Clasificación k-NN ----------------
with tab2:
    st.header("2. SA — Clasificación k-NN sobre breast-cancer.csv")
    st.markdown("Clasificación de tumores (benigno/maligno) con **k-Vecinos Más Cercanos** (k=6, distancia euclidiana, pesos por distancia).")
    df = load_breast()
    st.dataframe(df.head(5))
    X = df.drop(columns=["id", "diagnosis"]); y = df["diagnosis"].map({"M": 1, "B": 0})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=6, weights="distance", metric="euclidean"))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    show_metricas(metricas_clasificacion(yte, yp))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(yte, yp), display_labels=["Benigno", "Maligno"]).plot(cmap="Blues", ax=ax)
    ax.set_title("Matriz de confusión — k-NN")
    st.pyplot(fig)

# ---------------- 3 · NS Agrupación K-Means ----------------
with tab3:
    st.header("3. NS — Agrupación K-Means sobre samsung.csv")
    st.markdown("Agrupación de días de cotización de la acción Samsung (2008–2019) con **K-Means** (k=2 óptimo por silueta).")
    df = load_samsung()
    st.dataframe(df.head(5))
    df["retorno"] = df["Close"].pct_change() * 100
    df["vol_log"] = np.log1p(df["Volume"])
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df = df.dropna().reset_index(drop=True)
    features = ["Close", "Volume", "retorno", "vol_log", "ma5", "ma20"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    k = 2
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    sil = silhouette_score(X, km.labels_); db = davies_bouldin_score(X, km.labels_)
    c1, c2 = st.columns(2)
    c1.metric("Silueta", round(sil, 4)); c2.metric("Davies-Bouldin", round(db, 4))
    df["cluster"] = km.labels_
    fig, ax = plt.subplots()
    sc = ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=15, alpha=0.8)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", s=220, c="red", label="Centroides")
    ax.set_xlabel("Close (escalado)"); ax.set_ylabel("Volume (escalado)")
    ax.set_title(f"K-Means (k={k}) sobre la acción Samsung"); ax.legend()
    st.pyplot(fig)
    st.dataframe(df.groupby("cluster")[["Close", "Volume", "retorno"]].mean().round(2))

# ---------------- 4 · NS Reducción PCA ----------------
with tab4:
    st.header("4. NS — Reducción de dimensionalidad PCA sobre comprar_alquilar.csv")
    st.markdown("Análisis de Componentes Principales: comparación de los dos primeros componentes **sin** y **con** escalado, y justificación de n=7 componentes (95% de varianza).")
    df = load_comprar()
    st.dataframe(df.head(5))
    X = df.drop(columns=["comprar"]); y = df["comprar"]
    pca_sin = PCA(n_components=2, random_state=42).fit_transform(X)
    scaler = StandardScaler(); Xe = scaler.fit_transform(X)
    pca_con = PCA(n_components=2, random_state=42).fit(Xe)
    Xe2 = pca_con.transform(Xe)
    cola, colb = st.columns(2)
    with cola:
        fig, ax = plt.subplots()
        sc = ax.scatter(pca_sin[:, 0], pca_sin[:, 1], c=y, cmap="coolwarm", s=60, alpha=0.8)
        ax.set_title(f"Sin escalar (PC1 {0.99*100:.1f}% varianza)"); st.pyplot(fig)
    with colb:
        fig, ax = plt.subplots()
        sc = ax.scatter(Xe2[:, 0], Xe2[:, 1], c=y, cmap="coolwarm", s=60, alpha=0.8)
        ax.set_title(f"Con escalado (PC1 {pca_con.explained_variance_ratio_[0]*100:.1f}%)"); st.pyplot(fig)
    pca_full = PCA(random_state=42).fit(Xe)
    acum = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.argmax(acum >= 0.95) + 1)
    st.markdown(f"**Componentes elegidos:** {n95} (capturan {acum[n95-1]*100:.1f}% de la varianza, umbral ≥95%)")
    fig, ax = plt.subplots()
    ax.bar(range(1, len(acum) + 1), pca_full.explained_variance_ratio_, color="#2E86AB", label="Individual")
    ax.plot(range(1, len(acum) + 1), acum, marker="o", color="#A23B72", label="Acumulada")
    ax.axhline(0.95, ls="--", color="gray")
    ax.axvline(n95, ls="--", color="#3AA655")
    ax.set_xlabel("Componente"); ax.set_ylabel("Varianza"); ax.set_title("Scree plot y varianza acumulada")
    ax.legend(); st.pyplot(fig)

# ---------------- 5 · DE Clasificación Random Forest ----------------
with tab5:
    st.header("5. DE — Clasificación Random Forest sobre breast-cancer.csv")
    st.markdown("Alterno al k-NN de SA: **Random Forest** (100 árboles).")
    df = load_breast()
    X = df.drop(columns=["id", "diagnosis"]); y = df["diagnosis"].map({"M": 1, "B": 0})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    yp = rf.predict(Xte)
    show_metricas(metricas_clasificacion(yte, yp))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(yte, yp), display_labels=["Benigno", "Maligno"]).plot(cmap="Blues", ax=ax)
    ax.set_title("Matriz de confusión — Random Forest")
    st.pyplot(fig)

# ---------------- 6 · DE Regresión Random Forest ----------------
with tab6:
    st.header("6. DE — Regresión Random Forest sobre beisbol.csv")
    st.markdown("Alterno a la regresión lineal de SA: **Random Forest Regressor** (50 árboles).")
    df = load_beisbol()
    st.dataframe(df.head(10))
    X, y = df[["bateos"]], df["runs"]
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    yp = rf.predict(X)
    show_metricas(metricas_regresion(y, yp))
    plot_reales_vs_predichos(y, yp, "Valores reales vs predichos — Random Forest")

st.markdown("---")
st.caption("Evaluación U2 — SA, NS, DE y AU · Extracción de Conocimiento de Base de Datos · 2026")
