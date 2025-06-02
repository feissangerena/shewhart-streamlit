import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Shewhart Dashboard",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------
# 1. Funciones de lógica (casi idénticas al notebook)
# --------------------------------------------

def descargar_precios(ticker: str, interval: str, n_periodos: int) -> pd.DataFrame:
    """
    Descarga precios ajustados ('Adj Close') de Yahoo Finance.
    - ticker: símbolo (ej. 'AAPL').
    - interval: '1d', '1wk', '1mo' o '1y'.
    - n_periodos: número de unidades hacia atrás.
    Devuelve DataFrame con índice de fecha y columna 'Adj Close'.
    """
    hoy = datetime.today()
    multiplicador_dias = {'1d': 1, '1wk': 7, '1mo': 30, '1y': 365}
    dias_hacia_atras = n_periodos * multiplicador_dias.get(interval, 1) * 2
    fecha_inicio = hoy - timedelta(days=dias_hacia_atras)

    df = yf.download(
        ticker,
        start=fecha_inicio.strftime('%Y-%m-%d'),
        end=hoy.strftime('%Y-%m-%d'),
        interval=interval,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}. Verifica el ticker o intervalo.")

    # Usar 'Adj Close' si existe; si no, renombrar 'Close'
    if 'Adj Close' in df.columns:
        df_precio = df[['Adj Close']].copy()
    elif 'Close' in df.columns:
        df_precio = df[['Close']].copy()
        df_precio.rename(columns={'Close': 'Adj Close'}, inplace=True)
    else:
        raise KeyError("Ni 'Adj Close' ni 'Close' están en los datos descargados.")

    df_precio.dropna(inplace=True)
    return df_precio


def calcular_retornos_y_estadisticas(df: pd.DataFrame, tipo_retorno: str) -> pd.DataFrame:
    """
    A partir de df con columna 'Adj Close':
    - Si tipo_retorno == 'A', calcula retornos aritméticos: (P_t / P_{t-1}) - 1.
    - Si tipo_retorno == 'L', calcula retornos logarítmicos: ln(P_t / P_{t-1}).
    - Calcula media y desviación estándar global (ddof=0).
    - Genera columnas fijas: Media, STD, LSC_1σ, LIC_1σ, LSC_2σ, LIC_2σ, LSC_3σ, LIC_3σ, Z_score.
    Devuelve DataFrame con todas esas columnas.
    """
    df_ret = df.copy()
    if tipo_retorno == 'A':
        df_ret['Retorno'] = df_ret['Adj Close'].pct_change()
    else:  # 'L': retornos logarítmicos
        df_ret['Retorno'] = np.log(df_ret['Adj Close'] / df_ret['Adj Close'].shift(1))
    df_ret.dropna(inplace=True)

    media_val = df_ret['Retorno'].mean()
    std_val = df_ret['Retorno'].std(ddof=0)

    df_ret['Media'] = media_val
    df_ret['STD'] = std_val
    df_ret['LSC_1σ'] = media_val + std_val
    df_ret['LIC_1σ'] = media_val - std_val
    df_ret['LSC_2σ'] = media_val + 2 * std_val
    df_ret['LIC_2σ'] = media_val - 2 * std_val
    df_ret['LSC_3σ'] = media_val + 3 * std_val
    df_ret['LIC_3σ'] = media_val - 3 * std_val

    df_ret['Z_score'] = (df_ret['Retorno'] - media_val) / std_val
    return df_ret


def detectar_todas_las_reglas(df_shewhart: pd.DataFrame):
    """
    Aplica las 14 reglas de Shewhart sobre la columna 'Z_score' y 'Retorno'.
    Devuelve:
      - df_tabla_viol: DataFrame con columnas [Regla, Descripción, Nº Violaciones, Primeras Fechas].
      - media, std, dias_fuera_control (valores numéricos).
    """
    z_scores = df_shewhart['Z_score']
    retornos = df_shewhart['Retorno']
    fechas = df_shewhart.index

    # 1 & 2. Puntos fuera de ±3σ
    viol_1_arriba = z_scores[z_scores > 3].index.tolist()
    viol_2_abajo = z_scores[z_scores < -3].index.tolist()

    # 3 & 4. Nueve puntos consecutivos sobre/bajo la media
    idx_arriba_9, idx_abajo_9 = [], []
    serie_pos = (z_scores > 0).astype(int)
    c = 0
    for i in range(len(serie_pos)):
        if serie_pos.iloc[i] == 1:
            c += 1
        else:
            c = 0
        if c >= 9:
            idx_arriba_9.append(z_scores.index[i])
    serie_neg = (z_scores < 0).astype(int)
    c = 0
    for i in range(len(serie_neg)):
        if serie_neg.iloc[i] == 1:
            c += 1
        else:
            c = 0
        if c >= 9:
            idx_abajo_9.append(z_scores.index[i])

    # 5 & 6. Seis puntos consecutivos en tendencia creciente/decreciente
    idx_crec_6, idx_decrec_6 = [], []
    vals = retornos.values
    for i in range(len(vals) - 5):
        window = vals[i : i + 6]
        if np.all(np.diff(window) > 0):
            idx_crec_6.append(fechas[i + 5])
        if np.all(np.diff(window) < 0):
            idx_decrec_6.append(fechas[i + 5])

    # 7. Catorce alternancias de signo
    idx_alt_14 = []
    zs = z_scores.values
    for i in range(len(zs) - 13):
        w = zs[i : i + 14]
        alterna = True
        for j in range(1, 14):
            if w[j] == 0 or w[j-1] == 0 or w[j] * w[j-1] >= 0:
                alterna = False
                break
        if alterna:
            idx_alt_14.append(fechas[i + 13])

    # 8 & 9. Dos de tres puntos consecutivos fuera de ±2σ
    idx_2sobre2σ, idx_2bajo2σ = [], []
    for i in range(len(zs) - 2):
        w = zs[i : i + 3]
        if np.sum(w > 2) >= 2:
            idx_2sobre2σ.append(fechas[i + 2])
        if np.sum(w < -2) >= 2:
            idx_2bajo2σ.append(fechas[i + 2])

    # 10 & 11. Cuatro de cinco puntos fuera de ±1σ
    idx_4sobre1σ, idx_4bajo1σ = [], []
    for i in range(len(zs) - 4):
        w = zs[i : i + 5]
        if np.sum(w > 1) >= 4:
            idx_4sobre1σ.append(fechas[i + 4])
        if np.sum(w < -1) >= 4:
            idx_4bajo1σ.append(fechas[i + 4])

    # 12. Quince puntos consecutivos dentro de ±1σ
    idx_dentro15 = []
    for i in range(len(zs) - 14):
        w = zs[i : i + 15]
        if np.all(np.abs(w) < 1):
            idx_dentro15.append(fechas[i + 14])

    # 13 & 14. Ocho puntos consecutivos fuera de ±1σ
    idx_8sobre1σ, idx_8bajo1σ = [], []
    c = 0
    for i in range(len(zs)):
        if zs[i] > 1:
            c += 1
        else:
            c = 0
        if c >= 8:
            idx_8sobre1σ.append(fechas[i])
    c = 0
    for i in range(len(zs)):
        if zs[i] < -1:
            c += 1
        else:
            c = 0
        if c >= 8:
            idx_8bajo1σ.append(fechas[i])

    # Descripciones fijas para cada regla
    descripciones = {
        1:  "Un punto por encima de +3σ.",
        2:  "Un punto por debajo de −3σ.",
        3:  "Nueve puntos consecutivos por encima de la media.",
        4:  "Nueve puntos consecutivos por debajo de la media.",
        5:  "Seis puntos consecutivos en tendencia creciente.",
        6:  "Seis puntos consecutivos en tendencia decreciente.",
        7:  "Catorce puntos alternando signo (+, −, +, −...).",
        8:  "Dos de tres puntos consecutivos por encima de +2σ.",
        9:  "Dos de tres puntos consecutivos por debajo de −2σ.",
        10: "Cuatro de cinco puntos consecutivos por encima de +1σ.",
        11: "Cuatro de cinco puntos consecutivos por debajo de −1σ.",
        12: "Quince puntos consecutivos dentro de ±1σ.",
        13: "Ocho puntos consecutivos por encima de +1σ.",
        14: "Ocho puntos consecutivos por debajo de −1σ."
    }

    # Preparar la tabla de violaciones
    tabla_violaciones = []
    listas_por_regla = [
        idx for idx in [
            idx_2sobre2σ, idx_2bajo2σ,  # placeholder para índices 0 y 1
        ]
    ]  # no se usa directamente esa variable, pero ilustra la estructura.

    # Mapear cada número de regla a su lista correspondiente
    mapping = {
        1:  viol_1_arriba,
        2:  viol_2_abajo,
        3:  idx_arriba_9,
        4:  idx_abajo_9,
        5:  idx_crec_6,
        6:  idx_decrec_6,
        7:  idx_alt_14,
        8:  idx_2sobre2σ,
        9:  idx_2bajo2σ,
        10: idx_4sobre1σ,
        11: idx_4bajo1σ,
        12: idx_dentro15,
        13: idx_8sobre1σ,
        14: idx_8bajo1σ
    }

    for regla in range(1, 15):
        fechas_viol = mapping[regla]
        num_viol = len(fechas_viol)
        primeras = ", ".join(str(f.date()) for f in fechas_viol[:3]) if num_viol > 0 else "—"
        tabla_violaciones.append({
            "Regla": regla,
            "Descripción": descripciones[regla],
            "Nº Violaciones": num_viol,
            "Primeras Fechas": primeras
        })

    df_tabla_viol = pd.DataFrame(tabla_violaciones)

    # Estadísticas globales
    media = df_shewhart['Media'].iloc[0]
    std = df_shewhart['STD'].iloc[0]
    dias_fuera_control = len(set(
        viol_1_arriba + viol_2_abajo + idx_arriba_9 + idx_abajo_9 +
        idx_crec_6 + idx_decrec_6 + idx_alt_14 + idx_2sobre2σ + idx_2bajo2σ +
        idx_4sobre1σ + idx_4bajo1σ + idx_dentro15 + idx_8sobre1σ + idx_8bajo1σ
    ))

    return df_tabla_viol, media, std, dias_fuera_control


# --------------------------------------------
# 2. Interfaz de Streamlit
# --------------------------------------------

st.title("📈 Shewhart Dashboard (Móvil)")

st.markdown(
    """
    Bienvenido a la Calculadora de Retornos de Shewhart.
    Introduce un **ticker**, el **intervalo**, los **períodos** y el **tipo de retorno**.
    Pulsa **Calcular** para ver la tabla de violaciones y el gráfico de control.
    """
)

# --- Parámetros de entrada ---
col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    intervalo = st.selectbox("Intervalo", ["1d", "1wk", "1mo", "1y"])
with col2:
    periodos = st.number_input("Períodos", min_value=1, value=30, step=1)
    tipo = st.radio("Tipo de Retorno", ("Aritmético", "Logarítmico"))

calcular_btn = st.button("Calcular")

if calcular_btn:
    try:
        with st.spinner("Descargando datos y calculando..."):
            df_precios = descargar_precios(ticker, intervalo, int(periodos))
            df_shewhart = calcular_retornos_y_estadisticas(
                df_precios, 
                "A" if tipo == "Aritmético" else "L"
            )
            df_tabla_viol, media, std, dias_fuera_control = detectar_todas_las_reglas(df_shewhart)

        st.success("✅ Cálculo completado")

        # Mostrar tabla de violaciones
        st.subheader("📋 Tabla de Violaciones")
        st.table(df_tabla_viol)

        # Mostrar estadísticas globales
        st.subheader("📝 Resumen Estadístico")
        st.markdown(
            f"""
            - **Media de retornos**: `{media:.6f}`  
            - **Desviación estándar (σ)**: `{std:.6f}`  
            - **Días fuera de control**: `{dias_fuera_control}`
            """
        )

        # Gráfico de control
        st.subheader("📊 Gráfico de Control de Retornos")
        fig, ax = plt.subplots(figsize=(8, 4))
        fechas = df_shewhart.index
        retornos = df_shewhart["Retorno"]

        # Serie de retornos
        ax.plot(fechas, retornos, marker="o", linestyle="-", label="Retorno", color="blue")
        # Líneas de control ±1σ, ±2σ, ±3σ
        ax.axhline(media, color="orange", linestyle="--", label="Media")
        ax.axhline(media + std, color="green", linestyle="--", linewidth=0.8, label="+1σ")
        ax.axhline(media - std, color="green", linestyle="--", linewidth=0.8, label="-1σ")
        ax.axhline(media + 2 * std, color="purple", linestyle="--", linewidth=0.8, label="+2σ")
        ax.axhline(media - 2 * std, color="purple", linestyle="--", linewidth=0.8, label="-2σ")
        ax.axhline(media + 3 * std, color="red", linestyle="--", linewidth=1.0, label="+3σ")
        ax.axhline(media - 3 * std, color="red", linestyle="--", linewidth=1.0, label="-3σ")

        # Puntos violados (marcados en rojo)
        viol_indices = [
            fechas.get_loc(f)
            for f in df_tabla_viol["Primeras Fechas"].map(
                lambda s: [pd.to_datetime(date) for date in s.split(", ")] if s != "—" else []
            ).sum()
            if f in fechas
        ]
        if viol_indices:
            ax.scatter(
                fechas[viol_indices],
                retornos.iloc[viol_indices],
                color="red",
                edgecolor="black",
                label="Violaciones",
                zorder=5
            )

        ax.legend(loc="lower left", fontsize="small")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Retorno")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Botón para descargar CSV con toda la tabla de Shewhart
        csv_bytes = df_shewhart.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Descargar CSV completo",
            data=csv_bytes,
            file_name=f"{ticker}_shewhart.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
