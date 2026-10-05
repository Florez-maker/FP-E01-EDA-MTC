import io
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (KFold, GroupKFold, GroupShuffleSplit,
                                     cross_val_predict, train_test_split)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from PIL import Image

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Laboratorio Estadístico — FarmPrecision",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary":  "#1b60a7",
    "success":  "#2ca02c",
    "danger":   "#d62728",
    "warning":  "#F1C40F",
    "info":     "#17becf",
}

TEXT_COLS = {
    "finca", "lote", "departamento", "material", "variedad", "zona",
    "manejo", "formula", "foliar", "fuente",
    "textura_es", "textura_en", "prof", "ifc_clase",
    "tipo_de_analise_foliar", "tipo_analisis_foliar",
    "fertilizantes", "produto_comercial", "producto_comercial",
}

TARGET_PROD  = "ton_ha"
META_FFB_DEF = 25.0

EDAD_MAX_VALIDA = 60
SIEMBRA_MIN     = 1900

TON_HA_ALIASES = {
    "ton_ha": ["ton/ha", "tonha", "ton_ha", "rff/ha", "rff_ha",
               "rendimiento", "produccion", "producción"],
}
ANO_ALIASES = {
    "ano": ["año", "year", "anio", "campana", "campaña"],
}
CLIMA_ALIASES = {
    "pp_anual": ["pp anual", "pp_anual", "precipitacion anual",
                 "precipitacion", "precipitación anual"],
    "temp_promedio": ["temp promedio", "temp_promedio",
                      "temperatura promedio", "temperatura_promedio"],
    "humedad_relativa_promedio": ["humedad relativa promedio", "hr promedio",
                                  "hr_prom", "humedad_relativa"],
    "brillo_solar_promedio": ["brillo solar promedio", "brillo solar",
                              "brillo_solar"],
}
FOLIAR_ALIASES = {
    "fol_n":  ["n_1", "n_f", "n_fol", "n_foliar"],
    "fol_p":  ["p_1", "p_f", "p_fol", "p_foliar"],
    "fol_k":  ["k_1", "k_f", "k_fol", "k_foliar"],
    "fol_ca": ["ca_1", "ca_f", "ca_fol", "ca_foliar"],
    "fol_mg": ["mg_1", "mg_f", "mg_fol", "mg_foliar"],
    "fol_s":  ["s_1", "s_f", "s_fol", "s_foliar"],
    "fol_b":  ["b_1", "b_f", "b_fol", "b_foliar"],
    "fol_cu": ["cu_1", "cu_f", "cu_fol", "cu_foliar"],
    "fol_fe": ["fe_1", "fe_f", "fe_fol", "fe_foliar"],
    "fol_mn": ["mn_1", "mn_f", "mn_fol", "mn_foliar"],
    "fol_zn": ["zn_1", "zn_f", "zn_fol", "zn_foliar"],
    "fol_cl": ["cl_1", "cl_f", "cl_fol", "cl_foliar"],
}

PROD_BASE_COLS = [
    "ton_ha", "ton_lote", "kg_palma",
    "tch", "ffb", "prod_ffb", "rff_ha", "rendimiento", "produccion",
    "racimos", "n_racimos", "num_racimos", "racimos_ha",
    "peso_racimo", "peso_prom_racimo", "peso_racimos",
]
PROD_PATRONES = ("ton_ha", "tch", "ffb", "racim", "cosecha", "rend", "rff")
PROD_EXCLUI_PREFIJO = ("kg", "oferta_", "dist_opt_", "score_", "bma_", "fol_")

ESTRUCTURA_COLS = ["edad", "edad2", "densidad", "area", "n_palmas",
                   "ano", "siembra"]

ENDOGENAS_PROD = ("ton_lote", "kg_palma")

BLOQUES_FUTUROS = {
    "Suelo": [
        "ph", "cea", "ce", "mo", "p", "p_meh", "p_res", "p_rem", "p_total",
        "ca", "mg", "k", "na", "al", "cic", "cice", "acid_int",
        "sat_ca", "sat_mg", "sat_k", "sat_na", "sat_al", "sat_bases",
        "ca_mg", "mg_k", "ca_k", "ca_mg_k", "k_na", "psi", "ras", "fe_mn",
        "b", "cu", "fe", "mn", "zn", "s", "a", "l", "ar",
        "ifc", "indice_acidez", "indice_bases", "indice_sodicidad",
        "indice_micros", "indice_fertilidad_quimica",
    ],
    "Fertilización": [
        "kg_ha_n", "kg_n_ha", "kg_ha_p", "kg_p_ha", "kg_ha_k", "kg_k_ha",
        "kg_ha_ca", "kg_ca_ha", "kg_ha_mg", "kg_mg_ha", "kg_ha_s", "kg_s_ha",
        "kg_ha_b", "kg_b_ha", "kg_ha_zn", "kg_zn_ha", "kg_ha_cu", "kg_cu_ha",
        "kg_ha_fe", "kg_fe_ha", "kg_ha_mn", "kg_mn_ha",
        "kg_palma", "dosagem_kg_planta",
    ],
    "Foliar": list(FOLIAR_ALIASES.keys()),
    "Clima":  list(CLIMA_ALIASES.keys()),
}

# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }
    .kpi-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #1b60a7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #7A8899;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1C2B3A; line-height: 1.1; }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #0A3D62;
        border-bottom: 2px solid #1A6B3C;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }
    [data-testid="stSidebar"] { background: #F0F4F8; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.85rem;
    }
    .upload-zone {
        border: 2px dashed #1b60a7; border-radius: 10px;
        padding: 2rem; text-align: center; background: #f0f7ff;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. UTILIDADES DE NORMALIZACIÓN Y CARGA
# ══════════════════════════════════════════════════════════════════════════

_ACCENTOS = str.maketrans("áàäéèëíìïóòöúùüñçÁÀÄÉÈËÍÌÏÓÒÖÚÙÜÑÇ",
                          "aaaeeeiiiooouuuncAAAEEEIIIOOOUUUNC")


def _strip_acentos(s: str) -> str:
    return s.translate(_ACCENTOS)

try:
    from unidecode import unidecode as _unidecode
except ImportError:
    _unidecode = _strip_acentos


def sanitize_key(s) -> str:
    return (str(s).replace(" ", "_").replace("/", "_").replace("-", "_")
                  .replace(".", "_").replace("(", "").replace(")", ""))


def normalize_colname(col: str) -> str:
    col = _unidecode(str(col).strip()).lower()
    col = col.replace("%", "pct")
    col = col.replace("/", "_").replace("-", "_").replace("+", "_").replace(".", "_")
    col = re.sub(r"[()\[\]{}]", "", col)
    col = re.sub(r"[^a-z0-9_]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def normalizar_texto(value):
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        return np.nan
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "nat"}:
        return np.nan
    return _unidecode(value).upper()


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df.columns = [normalize_colname(c) for c in df.columns]

    duplicadas = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicadas:
        print(f"  ⚠️ Columnas duplicadas tras normalizar: {duplicadas}")
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(normalizar_texto)
    return df


def resolve_aliases(df: pd.DataFrame, alias_map: dict):
    df = df.copy()
    found = []
    cols_por_nombre_norm = {normalize_colname(c): c for c in df.columns}

    for canonical, aliases in alias_map.items():
        if canonical in df.columns:
            found.append(canonical)
            continue
        for alias in aliases:
            alias_norm = normalize_colname(alias)
            if alias_norm in cols_por_nombre_norm:
                original = cols_por_nombre_norm[alias_norm]
                df = df.rename(columns={original: canonical})
                found.append(canonical)
                break
    return df, found


def cargar_dataset_consolidado(file_bytes: bytes, file_name: str = "") -> pd.DataFrame:
    is_csv = isinstance(file_name, str) and file_name.lower().endswith(".csv")

    if is_csv:
        try:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=None,
                                 engine="python", dtype=object)
        except Exception:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=",", dtype=object)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]
    df = standardize_df(df_raw)

    # Alias robustos → nomenclatura canónica del ecosistema FarmPrecision
    df, tonha_det = resolve_aliases(df, TON_HA_ALIASES)
    df, ano_det   = resolve_aliases(df, ANO_ALIASES)
    df, _         = resolve_aliases(df, CLIMA_ALIASES)
    df, _         = resolve_aliases(df, FOLIAR_ALIASES)

    for col in df.columns:
        if col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  ✓ Consolidado estandarizado: {df.shape}")
    print(f"  ✓ Producción (ton_ha): {'sí' if TARGET_PROD in df.columns else 'NO'} "
          f"| detectado vía: {tonha_det}")
    return df


def limpiar_panel(df: pd.DataFrame):
    """v1.1 — Sanidad específica del panel lote×año.

    * Edad > EDAD_MAX_VALIDA → año de muestreo colado como edad (p. ej.
      registros POUSIO con 2021–2026): se trata como nulo para no romper la
      curva de producción.
    * Siembra < SIEMBRA_MIN (0) → inválida.
    * Construye edad2 (término cuadrático de la curva de producción).
    Devuelve (df, info) con los conteos para el banner de calidad.
    """
    info = {"edad_anomala": 0, "siembra_invalida": 0, "siembra_futura": 0}
    df = df.copy()

    if "edad" in df.columns:
        e = pd.to_numeric(df["edad"], errors="coerce")
        malas = e > EDAD_MAX_VALIDA
        info["edad_anomala"] = int(malas.sum())
        df["edad"] = e.mask(malas)
        df["edad2"] = df["edad"] ** 2

    if "siembra" in df.columns:
        s_ = pd.to_numeric(df["siembra"], errors="coerce")
        info["siembra_invalida"] = int((s_ < SIEMBRA_MIN).sum())
        df["siembra"] = s_.where(s_ >= SIEMBRA_MIN)
        if "ano" in df.columns:
            a = pd.to_numeric(df["ano"], errors="coerce")
            info["siembra_futura"] = int((s_ > a).sum())
    return df, info


def detectar_cols_produccion(df: pd.DataFrame) -> list:
    """Columnas de producción disponibles en el consolidado (numéricas)."""
    base = [c for c in PROD_BASE_COLS if c in df.columns]
    extras = [
        c for c in df.columns
        if any(p in c for p in PROD_PATRONES)
        and not c.startswith(PROD_EXCLUI_PREFIJO)
    ]
    return [c for c in list(dict.fromkeys(base + extras))
            if pd.api.types.is_numeric_dtype(df[c])]


def detectar_pool_eda(df: pd.DataFrame):
    """v1.1 — Pool completo del EDA: trío de producción + bloque estructural.

    Devuelve (pool_eda, prod_cols, estructura_cols). El bloque estructural
    (edad, edad², densidad, área, n_palmas, año, siembra) alimenta los tabs
    descriptivos y es la base de las predictoras exógenas del RF.
    """
    prod = detectar_cols_produccion(df)
    est = [c for c in ESTRUCTURA_COLS
           if c in df.columns and c not in prod
           and pd.api.types.is_numeric_dtype(df[c])]
    return prod + est, prod, est


def preparar_features(df: pd.DataFrame):
    """Clasifica columnas por bloque. La v1 usa PRODUCCIÓN + ESTRUCTURA, pero
    los demás bloques quedan calculados para reactivarse en la v2 sin refactor."""
    df = df.copy()
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(normalizar_texto)

    suelo = [c for c in BLOQUES_FUTUROS["Suelo"] if c in df.columns]
    fert  = [c for c in BLOQUES_FUTUROS["Fertilización"] if c in df.columns]
    fol   = [c for c in df.columns if c.startswith("fol_")]
    cli   = [c for c in BLOQUES_FUTUROS["Clima"] if c in df.columns]
    return df, suelo, fert, fol, cli


# ══════════════════════════════════════════════════════════════════════════
# 2. REFERENCIAS AGRONÓMICAS
# ══════════════════════════════════════════════════════════════════════════

RANGOS_REFERENCIA = {
    "ph":        (4.5, 5.0,  "—",        "Referentes calificación nutricional"),
    "mo":        (2.0, 4.0,  "%",        "Referentes calificación nutricional"),
    "cea":       (2.0, 4.0,  "dS/m",     "Referentes calificación nutricional"),
    "ce":        (2.0, 4.0,  "dS/m",     "Referentes calificación nutricional"),
    "cic":       (10.0, 20.0,"meq/100g", "Referentes calificación nutricional"),
    "k":         (0.2, 0.4,  "meq/100g", "Referentes calificación nutricional"),
    "p":         (15.0, 20.0,"ppm",      "P-Bray II — Referentes calificación"),
    "b":         (0.25, 0.50,"ppm",      "Referentes calificación nutricional"),
    "fe":        (15.0, 30.0,"ppm",      "Referentes calificación nutricional"),
    "cu":        (0.5, 1.5,  "ppm",      "Referentes calificación nutricional"),
    "mn":        (5.0, 10.0, "ppm",      "Referentes calificación nutricional"),
    "zn":        (1.0, 2.0,  "ppm",      "Referentes calificación nutricional"),
    "s":         (10.0, 15.0,"ppm",      "Referentes calificación nutricional"),
    "ca_mg":     (3.0, 5.0,  "ratio",    "Palma de aceite"),
    "mg_k":      (2.0, 4.0,  "ratio",    "Palma de aceite"),
    "ca_k":      (10.0, 14.0,"ratio",    "Palma de aceite"),
    "sat_ca":    (50.0, 60.0,"%",        "Palma de aceite"),
    "sat_mg":    (10.0, 15.0,"%",        "Palma de aceite"),
    "sat_k":     (4.0, 6.0,  "%",        "Palma de aceite"),
    "sat_al":    (0.0, 15.0, "%",        "Palma de aceite (<15%)"),
    "sat_bases": (60.0, 80.0,"%",        "V% — Palma de aceite (>60%)"),
}


def rango_optimo(var: str):
    ref = RANGOS_REFERENCIA.get(var)
    if ref is None:
        return None
    lo, hi, uni, src = ref
    return lo, hi, (lo + hi) / 2, uni, src


def referencia_meta(var: str, meta_val: float):
    """(meta, unidad, fuente) si a la variable le aplica la meta en ton/ha."""
    if meta_val is None:
        return None
    if any(p in var for p in ("ton_ha", "tch", "ffb", "rend", "rff")):
        return float(meta_val), "ton/ha", "Meta FFB — Gemelo Digital Nutricional"
    return None


# ══════════════════════════════════════════════════════════════════════════
# 3. ESTADÍSTICA Y GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════

def tabla_estadigrafos(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    filas = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        media = s.mean()
        de = s.std(ddof=1) if len(s) > 1 else np.nan
        q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
        ref = rango_optimo(c)
        cv = round(de / media * 100, 1) if (pd.notna(de) and media != 0) else np.nan
        filas.append({
            "Variable": c.upper(),
            "N": int(s.count()),
            "X (media)": round(media, 3),
            "Mn (mín)": round(s.min(), 3),
            "Mx (máx)": round(s.max(), 3),
            "DE": round(de, 3) if pd.notna(de) else np.nan,
            "CV %": cv,
            "Mediana": round(q2, 3),
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "Rango óptimo": f"{ref[0]}–{ref[1]} {ref[3]}" if ref else "sin referente",
            "% en rango": (round(s.between(ref[0], ref[1]).mean() * 100, 1)
                           if ref else np.nan),
        })
    return pd.DataFrame(filas)


def matriz_pearson(df: pd.DataFrame, cols: list, min_pares: int = 15):
    cols = [c for c in cols if c in df.columns]
    R = pd.DataFrame(np.nan, index=cols, columns=cols)
    P = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, a in enumerate(cols):
        # Diagonal fija (evita df[[a, a]] → selección duplicada → Series ambigua)
        R.loc[a, a], P.loc[a, a] = 1.0, 0.0
        for b in cols[i + 1:]:
            par = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(par) >= min_pares and par[a].nunique() > 1 and par[b].nunique() > 1:
                r, p = pearsonr(par[a], par[b])
                R.loc[a, b] = R.loc[b, a] = r
                P.loc[a, b] = P.loc[b, a] = p
    return R, P


def fig_heatmap_pearson(R: pd.DataFrame, P: pd.DataFrame, alpha: float,
                        solo_signif: bool):
    M = R.copy()
    if solo_signif:
        M = M.where(P <= alpha)
    fig = px.imshow(M, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    text_auto=".2f", aspect="auto",
                    labels=dict(x="", y="", color="r de Pearson"))
    fig.update_traces(textfont=dict(size=10))
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=max(460, 40 * len(M.columns) + 140),
                      template="plotly_white",
                      margin=dict(t=20, b=90, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def fig_histograma_optimo(serie: pd.Series, var: str, nbins: int = 30,
                          meta_val: float = None) -> go.Figure:
    s = pd.to_numeric(serie, errors="coerce").dropna()
    fig = go.Figure()
    if s.empty:
        return fig

    fig.add_trace(go.Histogram(
        x=s, nbinsx=nbins, histnorm="probability density",
        marker_color=COLORS["primary"], opacity=0.55, name="Frecuencia",
    ))

    if s.nunique() >= 3:
        try:
            kx = np.linspace(s.min(), s.max(), 300)
            fig.add_trace(go.Scatter(
                x=kx, y=gaussian_kde(s)(kx), mode="lines",
                line=dict(color=COLORS["primary"], width=2), name="KDE",
            ))
        except Exception:
            pass

    # Referencia gráfica: meta de producción (v1) o rango edáfico si existiera
    meta = referencia_meta(var, meta_val)
    if meta:
        mval, uni, src = meta
        fig.add_vline(x=mval, line_color=COLORS["success"], line_width=2.5,
                      annotation_text=f"Meta {mval:.1f} {uni}",
                      annotation_position="top right")
    else:
        ref = rango_optimo(var)
        if ref:
            lo, hi, opt, uni, src = ref
            fig.add_vrect(x0=lo, x1=hi, fillcolor=COLORS["success"], opacity=0.13,
                          line_width=0, annotation_text="Rango óptimo",
                          annotation_position="top left")
            fig.add_vline(x=opt, line_color=COLORS["success"], line_width=2.5,
                          annotation_text=f"Óptimo medio {opt:.2f} {uni}",
                          annotation_position="top right")

    fig.add_vline(x=float(s.mean()), line_dash="dash", line_color=COLORS["danger"],
                  annotation_text=f"μ={s.mean():.2f}",
                  annotation_position="bottom right")
    fig.add_vline(x=float(s.median()), line_dash="dot", line_color=COLORS["warning"],
                  annotation_text=f"Md={s.median():.2f}",
                  annotation_position="bottom left")

    fig.update_layout(height=360, template="plotly_white",
                      margin=dict(t=40, l=0, r=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title=var.upper(), yaxis_title="Densidad",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def fig_scatter_prod(df: pd.DataFrame, x: str, y: str = "ton_ha",
                     color_col=None, meta_val: float = None) -> go.Figure:
    d = df[[x, y] + ([color_col] if color_col and color_col in df.columns else [])].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if d.empty:
        return go.Figure()

    r, p = (pearsonr(d[x], d[y])
            if len(d) > 2 and d[x].nunique() > 1 else (np.nan, np.nan))

    fig = px.scatter(d, x=x, y=y, color=color_col,
                     opacity=0.75, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(marker=dict(size=8, line=dict(width=0.6, color="white")))

    # Recta OLS (solo si hay varianza en X)
    if d[x].nunique() > 1:
        try:
            b1, b0 = np.polyfit(d[x], d[y], 1)
            xs = np.linspace(d[x].min(), d[x].max(), 100)
            fig.add_trace(go.Scatter(x=xs, y=b0 + b1 * xs, mode="lines",
                                     line=dict(color=COLORS["danger"], width=2,
                                               dash="dash"),
                                     name=f"OLS: y = {b0:.2f} + {b1:.3f}·x"))
        except Exception:
            pass

    # Referencia: meta de producción en el eje X (si aplica)
    meta = referencia_meta(x, meta_val)
    if meta:
        fig.add_vline(x=meta[0], line_color=COLORS["success"], line_width=2,
                      annotation_text=f"Meta {meta[0]:.1f} {meta[1]}",
                      annotation_position="top left")
    else:
        ref = rango_optimo(x)
        if ref:
            fig.add_vrect(x0=ref[0], x1=ref[1], fillcolor=COLORS["success"],
                          opacity=0.12, line_width=0,
                          annotation_text="Óptimo", annotation_position="top left")

    fig.update_layout(
        title=f"{x.upper()} vs {y.upper()} — r = {r:.3f} (p = {p:.4f}, n = {len(d)})",
        height=440, template="plotly_white", margin=dict(t=50, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=x.upper(), yaxis_title="Producción (ton/ha)")
    return fig


def entrenar_rf(df, features, target="ton_ha", n_estimators=500, max_depth=12,
                min_samples_leaf=3, n_splits=5, umbral_leak=0.85, groups=None):
    """v1.1 — RF consciente del panel lote×año.

    Si `groups` (Serie con el lote, indexado como df) está disponible, el
    holdout y el K-fold se agrupan por lote (GroupShuffleSplit / GroupKFold):
    el mismo lote nunca queda repartido entre train y test, así el R² mide
    generalización a lotes nuevos y no memoria del lote.
    """
    d = df[[target] + list(features)].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=[target])
    feats_ok = [c for c in features if c in d.columns]
    X = d[feats_ok].loc[:, d[feats_ok].notna().any() & (d[feats_ok].nunique() > 1)]
    y = d[target].astype(float)

    # Control de fuga de información (|r| > umbral con el target)
    leaks = []
    for c in X.columns:
        par = pd.concat([X[c], y], axis=1).dropna()
        if len(par) >= 15 and par[c].nunique() > 1:
            r = par[c].corr(par[y.name])
            if pd.notna(r) and abs(r) > umbral_leak:
                leaks.append((c, r))
    if leaks:
        X = X.drop(columns=[c for c, _ in leaks])

    if X.empty or len(X) < 30:
        return None

    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                      columns=X.columns, index=X.index)

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf, max_features=0.8,
                               random_state=42, n_jobs=-1)

    g = groups.reindex(d.index) if groups is not None else None
    es_panel = g is not None and g.notna().sum() > 0 and g.nunique() >= n_splits

    if es_panel:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(Xi, y, groups=g))
        Xtr, Xte = Xi.iloc[tr_idx], Xi.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        rf.fit(Xtr, ytr)
        yhat = rf.predict(Xte)

        cv = GroupKFold(n_splits=n_splits)
        ycv = cross_val_predict(rf, Xi, y, cv=cv, groups=g, n_jobs=-1)
        cv_tipo = "panel"
    else:
        Xtr, Xte, ytr, yte = train_test_split(Xi, y, test_size=0.2, random_state=42)
        rf.fit(Xtr, ytr)
        yhat = rf.predict(Xte)

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        ycv = cross_val_predict(rf, Xi, y, cv=cv, n_jobs=-1)
        cv_tipo = "aleatoria"

    perm = permutation_importance(rf, Xte, yte, n_repeats=10,
                                  random_state=42, n_jobs=-1)

    mape = float(np.mean(np.abs((yte - yhat) / yte.replace(0, np.nan)).dropna()) * 100)

    return {
        "rf": rf, "X": Xi, "y": y, "yte": yte, "yhat": yhat, "ycv": ycv,
        "leaks": leaks, "cv_tipo": cv_tipo,
        "metricas": {
            "R2_test":  r2_score(yte, yhat),
            "R2_cv":    r2_score(y, ycv),
            "RMSE":     float(np.sqrt(mean_squared_error(yte, yhat))),
            "MAE":      float(mean_absolute_error(yte, yhat)),
            "MAPE_%":   mape,
        },
        "imp_gini": pd.DataFrame({"variable": Xi.columns,
                                  "importancia": rf.feature_importances_}
                                 ).sort_values("importancia", ascending=False),
        "imp_perm": pd.DataFrame({"variable": Xi.columns,
                                  "importancia": perm.importances_mean,
                                  "sd": perm.importances_std}
                                 ).sort_values("importancia", ascending=False),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. TABS — v1.1: producción + bloque estructural
# ══════════════════════════════════════════════════════════════════════════

def tab_estadigrafos(df, pool):
    st.markdown('<div class="section-title">1 · Estadígrafos descriptivos '
                '(producción + estructura)</div>', unsafe_allow_html=True)
    sel = st.multiselect("Variables:", pool,
                         default=pool[:min(12, len(pool))], key="eda_est_vars")
    if not sel:
        st.info("Selecciona al menos una variable."); return
    tab = tabla_estadigrafos(df, sel)
    st.dataframe(tab, use_container_width=True, hide_index=True)
    st.download_button("⬇ Descargar estadígrafos (CSV)",
                       tab.to_csv(index=False).encode("utf-8-sig"),
                       "estadigrafos.csv", "text/csv")
    st.caption("CV % = DE/X·100 · La meta de producción se ajusta con el slider "
               "superior (25 t FFB/ha por defecto).")


def tab_histogramas(df, pool, meta_val):
    st.markdown('<div class="section-title">2–3 · Histogramas de frecuencia y meta de producción</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox("Variable:", pool, key="eda_hist_var")
    nbins = c2.slider("Bins:", 10, 60, 30, key="eda_hist_bins")

    st.plotly_chart(fig_histograma_optimo(df[var], var, nbins, meta_val),
                    use_container_width=True, key=sanitize_key(f"eda_hist_{var}"))

    s = pd.to_numeric(df[var], errors="coerce").dropna()
    meta = referencia_meta(var, meta_val)
    if meta and not s.empty:
        mval, uni, src = meta
        c1, c2, c3 = st.columns(3)
        c1.metric("Media", f"{s.mean():.2f} {uni}")
        c2.metric("% registros ≥ meta", f"{(s >= mval).mean()*100:.1f} %")
        c3.metric("Brecha media (μ − meta)", f"{s.mean() - mval:+.2f} {uni}")
        st.table(pd.DataFrame([{"Variable": var.upper(), "Unidad": uni,
                                "Meta (ton/ha)": mval, "Fuente": src}]))
    else:
        st.caption(f"Sin meta de producción aplicable a `{var}` "
                   "(la meta se asigna a variables en ton/ha, TCH, FFB o RFF).")


def tab_pearson(df, pool):
    st.markdown('<div class="section-title">4 · Correlación de Pearson '
                '(producción vs estructura)</div>', unsafe_allow_html=True)
    sel = st.multiselect("Variables (se añade el target automáticamente):", pool,
                         default=pool[:min(10, len(pool))], key="eda_corr_vars")
    c1, c2 = st.columns(2)
    alpha = c1.select_slider("Nivel de significancia α:",
                             [0.001, 0.01, 0.05, 0.10], value=0.05)
    solo = c2.checkbox("Mostrar solo correlaciones significativas", value=True)

    cols = ([TARGET_PROD] if TARGET_PROD in df.columns else []) + \
           [c for c in sel if c != TARGET_PROD]
    if len(cols) < 2:
        st.info("Selecciona al menos dos variables."); return

    R, P = matriz_pearson(df, cols)
    st.plotly_chart(fig_heatmap_pearson(R, P, alpha, solo),
                    use_container_width=True, key="eda_pearson_heat")

    if TARGET_PROD in R.columns:
        rank = pd.DataFrame({"Variable": R.index, "r": R[TARGET_PROD],
                             "p_valor": P[TARGET_PROD]})
        rank = rank[rank["Variable"] != TARGET_PROD].dropna(subset=["r"])
        rank["|r|"] = rank["r"].abs()
        rank["Significativa"] = np.where(rank["p_valor"] <= alpha, "Sí", "No")
        st.dataframe(rank.sort_values("|r|", ascending=False).round(4),
                     use_container_width=True, hide_index=True)
        st.caption("Nota: `ton_lote` y `kg_palma` correlacionan fuerte con `ton_ha` "
                   "por construcción (son re-expresiones del mismo yield) — úsalas "
                   "solo en lectura descriptiva, nunca como predictoras.")


def tab_scatter(df, pool, meta_val):
    st.markdown('<div class="section-title">5 · Dispersión X vs Producción (ton_ha)</div>',
                unsafe_allow_html=True)
    if TARGET_PROD not in df.columns:
        st.info(f"No existe la columna {TARGET_PROD}."); return
    xs_opts = [c for c in pool if c != TARGET_PROD]
    c1, c2, c3 = st.columns([2, 1, 1])
    xs = c1.multiselect("Variables X (a voluntad):", xs_opts,
                        default=xs_opts[:min(2, len(xs_opts))], key="eda_scatter_x")
    grupos = [c for c in ["finca", "zona", "departamento", "variedad",
                          "material", "lote"] if c in df.columns]
    color = c2.selectbox("Colorear por:", ["(ninguno)"] + grupos,
                         key="eda_scatter_color")
    ncols = c3.radio("Columnas:", [1, 2], index=1, horizontal=True,
                     key="eda_scatter_ncols")
    color = None if color == "(ninguno)" else color

    if not xs:
        st.info("Selecciona al menos una variable X."); return
    cols_ui = st.columns(ncols)
    for i, x in enumerate(xs):
        with cols_ui[i % ncols]:
            st.plotly_chart(fig_scatter_prod(df, x, TARGET_PROD, color, meta_val),
                            use_container_width=True,
                            key=sanitize_key(f"eda_sc_{x}_{color}"))


def tab_rf(df, pool):
    st.markdown('<div class="section-title">6 · Random Forest — drivers de la producción</div>',
                unsafe_allow_html=True)
    if TARGET_PROD not in df.columns:
        st.info(f"No existe la columna {TARGET_PROD}."); return

    st.caption("Validación consciente del panel lote×año: el holdout y el CV se "
               "agrupan por lote (GroupShuffleSplit / GroupKFold), de modo que el R² "
               "refleja la capacidad de generalizar a lotes nuevos, no memoria del lote.")

    with st.expander("⚙️ Hiperparámetros y validación", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        n_est = c1.slider("n_estimators", 100, 1000, 500, 100)
        depth = c2.slider("max_depth", 3, 25, 12)
        leaf = c3.slider("min_samples_leaf", 1, 10, 3)
        folds = c4.slider("K folds", 3, 10, 5)
        usar_ano = st.checkbox(
            "Incluir 'Año' como predictora",
            value=True, key="eda_rf_ano",
            help="El año captura la señal climática interanual (drivers explicativos). "
                 "Desactívalo si el modelo se usará para pronosticar años futuros.",
        )

    feats_opts = [c for c in pool if c != TARGET_PROD]
    base_feats = [c for c in ["edad", "edad2", "densidad", "area", "n_palmas"]
                  if c in feats_opts]
    if usar_ano and "ano" in feats_opts:
        base_feats.append("ano")
    feats = st.multiselect("Variables predictoras (estructurales por defecto):",
                           feats_opts, default=base_feats, key="eda_rf_feats")
    if not feats:
        st.info("Selecciona al menos una variable predictora."); return

    endo = [c for c in feats if c in ENDOGENAS_PROD]
    if endo:
        st.warning("Endógenas seleccionadas (" + ", ".join(endo) + "): son "
                   "re-expresiones del target (ton/lote = ton/ha·área; "
                   "kg/palma = ton/lote·1000/N_palmas) y sobre-ajustan el modelo.")
        if not st.checkbox("Mantenerlas de todos modos (solo diagnóstico)",
                           value=False, key="eda_rf_endo"):
            feats = [c for c in feats if c not in ENDOGENAS_PROD]
    if not feats:
        st.info("Sin predictoras válidas tras excluir endógenas."); return

    groups = df["lote"] if "lote" in df.columns else None
    res = entrenar_rf(df, feats, TARGET_PROD, n_est, depth, leaf, folds,
                      groups=groups)
    if res is None:
        st.warning("Datos insuficientes (mínimo 30 registros con target válido).")
        return

    if res["leaks"]:
        st.info("Excluidas por |r| > 0.85 con el target: " +
                ", ".join(f"{c} (r={r:+.2f})" for c, r in res["leaks"]))

    m = res["metricas"]
    cv_lab = (f"R² CV-{folds} · "
              + ("GroupKFold (lote)" if res["cv_tipo"] == "panel"
                 else "aleatoria"))
    k = st.columns(5)
    for col, (lab, val) in zip(k, [("R² test", f"{m['R2_test']:.3f}"),
                                   (cv_lab, f"{m['R2_cv']:.3f}"),
                                   ("RMSE", f"{m['RMSE']:.2f}"),
                                   ("MAE", f"{m['MAE']:.2f}"),
                                   ("MAPE", f"{m['MAPE_%']:.1f} %")]):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{lab}</div>'
                     f'<div class="kpi-value">{val}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        modo = st.radio("Importancia:", ["Permutación", "Gini"],
                        horizontal=True, key="eda_rf_imp")
        imp = (res["imp_perm"] if modo == "Permutación" else res["imp_gini"]).head(20)
        fig = px.bar(imp.sort_values("importancia"), x="importancia", y="variable",
                     orientation="h", color="importancia",
                     color_continuous_scale="Greens")
        fig.update_layout(height=560, template="plotly_white",
                          coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="eda_rf_imp_plot")
    with c2:
        yte, yhat = res["yte"], res["yhat"]
        f = go.Figure()
        f.add_trace(go.Scatter(x=yte, y=yhat, mode="markers", name="Test",
                               marker=dict(color=COLORS["primary"], size=7,
                                           opacity=0.65)))
        lo, hi = float(min(yte.min(), yhat.min())), float(max(yte.max(), yhat.max()))
        f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                               line=dict(color=COLORS["danger"], dash="dash")))
        f.update_layout(title=f"Predicho vs Real — R² = {m['R2_test']:.3f}",
                        xaxis_title="Real (ton/ha)", yaxis_title="Predicho (ton/ha)",
                        height=560, template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(f, use_container_width=True, key="eda_rf_pred")

    st.markdown('<div class="section-title">Efecto agronómico estimado (P25 vs P75)</div>',
                unsafe_allow_html=True)
    top5 = res["imp_perm"]["variable"].head(5).tolist()
    cols5 = st.columns(len(top5))
    for i, v in enumerate(top5):
        s = pd.to_numeric(df[v], errors="coerce")
        if not s.notna().any():
            continue
        p25, p75 = s.quantile([0.25, 0.75])
        mu_lo = df.loc[s <= p25, TARGET_PROD].mean()
        mu_hi = df.loc[s >= p75, TARGET_PROD].mean()
        delta = mu_hi - mu_lo
        color = COLORS["success"] if delta > 0 else COLORS["danger"]
        cols5[i].markdown(
            f'<div class="kpi-card" style="border-left:4px solid {color}">'
            f'<div style="font-weight:700">{v.upper()}</div>'
            f'<div style="font-size:.82rem;margin-top:.3rem">'
            f'P25 ≤ {p25:.2f}: <b>{mu_lo:.2f}</b> ton/ha<br>'
            f'P75 ≥ {p75:.2f}: <b>{mu_hi:.2f}</b> ton/ha<br>'
            f'<span style="color:{color};font-weight:700">Δ = {delta:+.2f} ton/ha</span>'
            f'</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 5. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════


def render_sidebar():
    with st.sidebar:
        try:
            img = Image.open("logo_sidebar.png")
            st.image(img, width=260)
        except Exception:
            st.markdown("## 🌴 FarmPrecision")

        st.markdown("---")
        st.markdown("### 📂 Cargar datos")
        uploaded = st.file_uploader(
            "Sube tu archivo Excel o CSV",
            type=["xlsx", "xls", "csv"],
            help="Panel lote×año: finca, lote, año, edad, material y producción",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Columnas requeridas")
        st.markdown("""
        | **Variable** | **Nombre interno** | **Variantes aceptadas** |
        |:-------------|:-------------------|:-------------------------|
        | **Finca** | `finca` | finca, farm, hacienda |
        | **Lote** | `lote` | lote, lot, parcela |
        | **Año** | `ano` | año, ano, year, campaña |
        | **Zona** | `zona` | zona, zone |
        | **Variedad** | `variedad` | variedad, variety |
        | **Material** | `material` | material, genotipo |
        | **Edad** | `edad` | edad, age |
        | **Producción** | `ton_ha` | ton/ha, rff/ha, rendimiento, TCH, FFB |
        | **Prod. lote** | `ton_lote` | ton/lote |
        | **Prod. palma** | `kg_palma` | kg/palma |
        """)

        st.caption(
            "El sistema normaliza automáticamente mayúsculas, minúsculas, "
            "acentos, espacios y caracteres especiales."
        )
        st.markdown("---")
    return uploaded


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════


def main():
    uploaded = render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>Laboratorio Estadístico — Drivers de Productividad</h1>
        <p>FP-E01-EDA-MTC · v1.1 panel lote×año · Producción (ton/ha, ton/lote, kg/palma) + estructura · Estadígrafos · Meta · Pearson · Dispersión · Random Forest</p>
    </div>""", unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Sube tu archivo Excel o CSV para comenzar</h3>
            <p><strong>Formatos soportados:</strong> .xlsx · .xls · .csv</p>
            <p><strong>Columnas principales:</strong> Finca · Lote · Edad · ton/ha</p>
            <p><strong>Opcionales:</strong> Año · Siembra · Área · Densidad · N° de palmas ·
            variables de suelo · foliares · clima · fertilización</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 ¿Qué puedes analizar?")

        c1, c2, c3, c4 = st.columns(4)
        c1.info(
            "**Estadística descriptiva**\n\n"
            "Media, mediana, mínimo, máximo, desviación estándar, "
            "coeficiente de variación y cuartiles."
        )
        c2.info(
            "**Producción y estructura**\n\n"
            "Analiza ton/ha y variables como edad, densidad, área, "
            "número de palmas y año."
        )
        c3.info(
            "**Relaciones entre variables**\n\n"
            "Correlaciones de Pearson, distribuciones, histogramas "
            "y relaciones entre variables productivas."
        )
        c4.info(
            "**Modelamiento**\n\n"
            "Explora relaciones entre las variables del cultivo y la "
            "producción mediante análisis estadístico y modelos predictivos."
        )

        return

    with st.spinner("⏳ Procesando..."):
        data = cargar_dataset_consolidado(uploaded.read(), file_name=uploaded.name)
        data, info_cal = limpiar_panel(data)
        data, SUELO, FERT, FOLIAR, CLIMA = preparar_features(data)

    with st.expander("🩺 Control de calidad del panel (automático)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{len(data):,}")
        c2.metric("Lotes", f"{data['lote'].nunique():,}"
                  if "lote" in data.columns else "—")
        c3.metric("Años", f"{pd.to_numeric(data['ano'], errors='coerce').nunique()}"
                  if "ano" in data.columns else "—")
        c4.metric("ton_ha nulas", f"{data[TARGET_PROD].isna().sum():,}"
                  if TARGET_PROD in data.columns else "—")
        st.caption(
            f"Edad inválida tratada como nula (> {EDAD_MAX_VALIDA} — p. ej. POUSIO "
            f"con año de muestreo colado): **{info_cal.get('edad_anomala', 0):,}** filas · "
            f"Siembra inválida (< {SIEMBRA_MIN}): **{info_cal.get('siembra_invalida', 0):,}** · "
            f"Siembra futura (Siembra > Año — resiembra planificada): "
            f"**{info_cal.get('siembra_futura', 0):,}** filas. "
            "Edad = 0 se conserva (indica ciclo previo a resiembra; "
            "confirmar semántica con el origen de datos).")

    with st.sidebar:
        st.markdown("### 🔎 Filtros")
        fincas_all = (sorted(data["finca"].dropna().unique().tolist())
                      if "finca" in data.columns else [])
        zonas_all = (sorted(data["zona"].dropna().unique().tolist())
                     if "zona" in data.columns else [])
        depts_all = (sorted(data["departamento"].dropna().unique().tolist())
                     if "departamento" in data.columns else [])
        anos_all = (sorted(pd.to_numeric(data["ano"], errors="coerce")
                           .dropna().unique().astype(int).tolist())
                    if "ano" in data.columns else [])
        vars_all = (sorted(data["variedad"].dropna().unique().tolist())
                    if "variedad" in data.columns else [])
        mats_all = (sorted(data["material"].dropna().unique().tolist())
                    if "material" in data.columns else [])

        finca_sel = st.multiselect("Finca:", fincas_all, default=fincas_all,
                                   key="f_finca")
        zona_sel = dept_sel = var_sel = None
        incluir_parcial = True
        if zonas_all:
            zona_sel = st.multiselect("Zona:", zonas_all, default=zonas_all,
                                      key="f_zona")
        if depts_all:
            dept_sel = st.multiselect("Departamento:", depts_all, default=depts_all,
                                      key="f_departamento")
        año_sel = st.multiselect("Año:", anos_all, default=anos_all, key="f_ano")
        if anos_all:
            incluir_parcial = st.checkbox(
                f"Incluir {anos_all[-1]} (año en curso, parcial)",
                value=False, key="f_parcial",
                help="El último año del panel suele venir con cosecha incompleta; "
                     "inclúyelo solo si analizas el acumulado parcial.")
        if vars_all:
            var_sel = st.multiselect("Variedad:", vars_all, default=vars_all,
                                     key="f_var")
        mat_sel = st.multiselect("Material:", mats_all, default=mats_all,
                                 key="f_mat")
        lote_q = st.text_input("Buscar Lote:", "", key="f_lote")

    df = data.copy()
    if finca_sel and "finca" in df.columns:
        df = df[df["finca"].isin(finca_sel)]
    if zona_sel is not None and "zona" in df.columns:
        df = df[df["zona"].isin(zona_sel)]
    if dept_sel is not None and "departamento" in df.columns:
        df = df[df["departamento"].isin(dept_sel)]
    if año_sel and "ano" in df.columns:
        df = df[df["ano"].isin(año_sel)]
    if (not incluir_parcial) and "ano" in df.columns and anos_all:
        df = df[df["ano"] < anos_all[-1]]
    if var_sel is not None and "variedad" in df.columns:
        df = df[df["variedad"].isin(var_sel)]
    if mat_sel and "material" in df.columns:
        df = df[df["material"].isin(mat_sel)]
    if lote_q.strip() and "lote" in df.columns:
        df = df[df["lote"].astype(str).str.contains(lote_q.strip(), case=False, na=False)]

    st.sidebar.write(f"Registros filtrados: **{len(df):,}**")

    if df.empty:
        st.warning("Sin datos con los filtros actuales.")
        return

    POOL_EDA, PROD_POOL, ESTRUCT = detectar_pool_eda(df)
    if not PROD_POOL:
        st.warning("No se detectaron variables de producción numéricas "
                   "(ton_ha, ton/lote, kg/palma, tch, ffb...).")
        st.caption("Columnas numéricas disponibles: " +
                   ", ".join(df.select_dtypes(include=[np.number]).columns[:40]))
        return
    if TARGET_PROD not in df.columns:
        st.warning(f"No se encontró la columna objetivo `{TARGET_PROD}` (ton/ha).")
        return

    meta_val = st.slider("Meta de producción — FFB (ton/ha):",
                         10.0, 60.0, META_FFB_DEF, 0.5, key="eda_meta_ffb")

    tab_names = ["📐 Estadígrafos", "📊 Histogramas", "🔗 Pearson", "✳️ Scatter vs Producción", "🌲 Random Forest"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_estadigrafos(df, POOL_EDA)
    with tabs[1]:
        tab_histogramas(df, POOL_EDA, meta_val)
    with tabs[2]:
        tab_pearson(df, POOL_EDA)
    with tabs[3]:
        tab_scatter(df, POOL_EDA, meta_val)
    with tabs[4]:
        tab_rf(df, POOL_EDA)


if __name__ == "__main__":
    main()