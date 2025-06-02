import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from io import BytesIO

# --------------------------------------------------------
# 0. Configuración de la página Streamlit (icono + título + diseño)
# --------------------------------------------------------
st.set_page_config(
    page_title="Shewhart Dashboard",
    page_icon="📊",           # Icono de la pestaña
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Pantalla inicial con icono, título y leyenda, centrado
st.markdown(
    """
    <div style="text-align: center; margin-top: 30px;">
        <span style="font-size: 60px;">📊</span><br>
        <span style="font-size: 36px; font-weight: bold;">Shewhart Dashboard (Móvil)</span><br>
        <span style="font-size: 16px; color: #555;">
            🚀 Esta aplicación está alimentada por la API de Yahoo Finance.<br>
            Utiliza precios históricos para calcular retornos y aplicar las reglas de Shewhart.
        </span>
    </div>
    <hr style="margin-top: 20px; margin-bottom: 20px; border: none; height: 1px; background-color: #ddd;">
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# 1. Funciones auxiliares
# --------------------------------------------------------
def obtener_nombre_empresa(ticker: str) -> str:
    """
    Intenta extraer el nombre largo de la empresa desde yfinance.
    Si falla, devuelve el ticker.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except Exception:
        return ticker

def descargar_precios(ticker: str, interval: str, n_periodos: int) -> pd.DataFrame:
    """
    Descarga precios ajustados ('Adj Close') de Yahoo Finance.
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
      - Retornos aritméticos (A) o logarítmicos (L).
      - Calcula media y desviación estándar global.
      - Genera columnas: Media, STD, LSC_1σ, LIC_1σ, LSC_2σ, LIC_2σ, LSC_3σ, LIC_3σ, Z_score.
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
    Aplica las 14 reglas de Shewhart sobre 'Z_score' y 'Retorno'.
    Devuelve:
      - df_tabla_viol: DataFrame [Regla, Descripción, Nº Violaciones, Primeras Fechas]
      - df_detalle_viol: DataFrame [Fecha, Regla, Retorno, Z_score]
      - media, std, dias_fuera_control
    """
    z_scores = df_shewhart['Z_score']
    retornos = df_shewhart['Retorno']
    fechas = df_shewhart.index

    # 1. Punto fuera de +3σ
    viol_1_arriba = z_scores[z_scores > 3].index.tolist()
    # 2. Punto fuera de -3σ
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
            if w[j] == 0 or w[j - 1] == 0 or w[j] * w[j - 1] >= 0:
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

    # 10 & 11. Cuatro de cinco puntos consecutivos fuera de ±1σ
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

    tabla_violaciones = []
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
        primeras = (
            ", ".join(str(f.date()) for f in fechas_viol[:3]) 
            if num_viol > 0 else "—"
        )
        tabla_violaciones.append({
            "Regla": regla,
            "Descripción": descripciones[regla],
            "Nº Violaciones": num_viol,
            "Primeras Fechas": primeras
        })
    df_tabla_viol = pd.DataFrame(tabla_violaciones)

    detalle = []
    for regla, fechas_list in mapping.items():
        for fecha in fechas_list:
            fila = df_shewhart.loc[fecha]
            detalle.append({
                "Fecha": fecha.date(),
                "Regla": regla,
                "Retorno": float(fila["Retorno"]),
                "Z_score": float(fila["Z_score"])
            })
    df_detalle_viol = pd.DataFrame(detalle).sort_values("Fecha")

    media = df_shewhart['Media'].iloc[0]
    std = df_shewhart['STD'].iloc[0]
    dias_fuera_control = len(set(
        viol_1_arriba + viol_2_abajo + idx_arriba_9 + idx_abajo_9 +
        idx_crec_6 + idx_decrec_6 + idx_alt_14 + idx_2sobre2σ + idx_2bajo2σ +
        idx_4sobre1σ + idx_4bajo1σ + idx_dentro15 + idx_8sobre1σ + idx_8bajo1σ
    ))

    return df_tabla_viol, df_detalle_viol, media, std, dias_fuera_control

def generar_excel(df_precios: pd.DataFrame,
                   df_shewhart: pd.DataFrame,
                   df_tabla_viol: pd.DataFrame,
                   nombre_empresa: str,
                   ticker: str,
                   intervalo: str,
                   periodos: int,
                   tipo_retorno: str,
                   media: float,
                   std: float,
                   dias_fuera_control: int) -> BytesIO:
    """
    Genera un archivo Excel en memoria con tres pestañas:
      - 'Precios': precios ajustados.
      - 'Retornos': retornos, estadísticas y límites.
      - 'InformeShewhart': resumen estadístico + tabla de violaciones + gráfico.
    Devuelve BytesIO listo para descargar.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # 1) Hoja 'Precios'
        df_precios.to_excel(writer, sheet_name="Precios", index=True)

        # 2) Hoja 'Retornos'
        df_shewhart.to_excel(writer, sheet_name="Retornos", index=True)

        # 3) Hoja 'InformeShewhart'
        df_resumen = pd.DataFrame({
            "Campo": ["Nombre Empresa", "Ticker", "Intervalo", "Períodos analizados",
                      "Tipo de Retorno", "Media de retornos", "Desviación estándar σ",
                      "Días fuera de control"],
            "Valor": [nombre_empresa, ticker, intervalo, f"{periodos}",
                      tipo_retorno, f"{media:.6f}", f"{std:.6f}", f"{dias_fuera_control}"]
        })
        df_resumen.to_excel(writer, sheet_name="InformeShewhart", index=False, startrow=0)

        startrow = len(df_resumen.index) + 2
        df_tabla_viol.to_excel(writer, sheet_name="InformeShewhart", index=False, startrow=startrow)

        workbook  = writer.book
        worksheet = writer.sheets["InformeShewhart"]

        # Gráfico en Matplotlib
        fig, ax = plt.subplots(figsize=(6, 3))
        fechas = df_shewhart.index
        retornos = df_shewhart["Retorno"]
        ax.plot(fechas, retornos, marker="o", linestyle="-", label="Retorno", color="blue")
        ax.axhline(media, color="orange", linestyle="--", label="Media")
        ax.axhline(media + std, color="green", linestyle="--", linewidth=0.8, label="+1σ")
        ax.axhline(media - std, color="green", linestyle="--", linewidth=0.8, label="-1σ")
        ax.axhline(media + 2 * std, color="purple", linestyle="--", linewidth=0.8, label="+2σ")
        ax.axhline(media - 2 * std, color="purple", linestyle="--", linewidth=0.8, label="-2σ")
        ax.axhline(media + 3 * std, color="red", linestyle="--", linewidth=1.0, label="+3σ")
        ax.axhline(media - 3 * std, color="red", linestyle="--", linewidth=1.0, label="-3σ")

        viols = []
        for _, row in df_detalle_viol.iterrows():
            viols.append(pd.to_datetime(row["Fecha"]))
        viol_indices = [fechas.get_loc(f) for f in viols if f in fechas]
        if viol_indices:
            ax.scatter(
                fechas[viol_indices],
                retornos.iloc[viol_indices],
                color="red",
                edgecolor="black",
                label="Violación",
                zorder=5
            )

        ax.legend(loc="lower left", fontsize="small")
        ax.set_title(f"Gráfico de Control – {ticker}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Retorno")
        plt.xticks(rotation=45)
        plt.tight_layout()

        img_data = BytesIO()
        fig.savefig(img_data, format="png")
        plt.close(fig)
        img_data.seek(0)

        img_row = startrow + len(df_tabla_viol.index) + 2
        worksheet.insert_image(img_row, 0, "gráfico.png", {'image_data': img_data})

    output.seek(0)
    return output

# --------------------------------------------------------
# 2. Interfaz de Streamlit
# --------------------------------------------------------

# 2.1. Expander para parámetros (desplegado por defecto)
with st.expander("⚙️ Parámetros de Análisis", expanded=True):
    ticker = st.text_input("📌 Ticker", value="AAPL").strip().upper()
    intervalo = st.selectbox("⏱ Intervalo", ["1d", "1wk", "1mo", "1y"])
    periodos = st.number_input("🔢 Períodos (hacia atrás)", min_value=1, value=30, step=1)
    tipo = st.radio("📊 Tipo de Retorno", ("Aritmético", "Logarítmico"))

    # Checkbox para elegir cada σ
    mostrar_1sigma = st.checkbox("Mostrar ±1σ", value=True)
    mostrar_2sigma = st.checkbox("Mostrar ±2σ", value=True)
    mostrar_3sigma = st.checkbox("Mostrar ±3σ", value=True)

    calcular_btn = st.button("▶ Calcular")

# 2.2. Mensaje inicial si aún no se presionó "Calcular"
if not calcular_btn:
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:18px; margin-top:40px;'>"
        "🔍 Por favor, completa los parámetros y presiona **Calcular** para ver los resultados."
        "</div>",
        unsafe_allow_html=True
    )
    st.stop()

# --------------------------------------------------------
# 3. Lógica principal (al presionar "Calcular")
# --------------------------------------------------------
try:
    nombre_empresa = obtener_nombre_empresa(ticker)
    df_precios = descargar_precios(ticker, intervalo, periodos)
    df_shewhart = calcular_retornos_y_estadisticas(
        df_precios,
        "A" if tipo == "Aritmético" else "L"
    )
    df_tabla_viol, df_detalle_viol, media, std, dias_fuera_control = detectar_todas_las_reglas(df_shewhart)

    st.success("✅ Cálculo completado")

    # --------------------------------------------------------
    # 4. Resumen Estadístico (primero)
    # --------------------------------------------------------
    st.subheader("📝 Resumen Estadístico")
    st.markdown(
        f"""
        - **Nombre Empresa:** `{nombre_empresa}`  
        - **Ticker:** `{ticker}`  
        - **Intervalo:** `{intervalo}`  
        - **Períodos analizados:** `{periodos}`  
        - **Tipo de Retorno:** `{tipo}`  
        - **Media de retornos:** `{media:.6f}`  
        - **Desviación estándar (σ):** `{std:.6f}`  
        - **Días fuera de control:** `{dias_fuera_control}`
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # 5. Tabla de Violaciones (resumida)
    # --------------------------------------------------------
    st.subheader("📋 Tabla de Violaciones")
    html_tabla = df_tabla_viol.to_html(index=False)
    html_tabla = (
        html_tabla
        .replace("<table", '<table style="text-align: left; font-size: 12px; border-collapse: collapse; width:100%;">')
        .replace("<th>", '<th style="text-align: left; padding: 4px; font-size: 12px;">')
        .replace("<td>", '<td style="text-align: left; padding: 4px; font-size: 12px;">')
    )
    st.markdown(html_tabla, unsafe_allow_html=True)

    # --------------------------------------------------------
    # 6. Detalle de Violaciones (opcional, en expander)
    # --------------------------------------------------------
    with st.expander("🔍 Detalle de Violaciones (cada punto)", expanded=False):
        if df_detalle_viol.empty:
            st.write("No se encontraron violaciones detalladas.")
        else:
            html_detalle = df_detalle_viol.to_html(index=False)
            html_detalle = (
                html_detalle
                .replace("<table", '<table style="text-align: left; font-size: 12px; border-collapse: collapse; width:100%;">')
                .replace("<th>", '<th style="text-align: left; padding: 4px; font-size: 12px;">')
                .replace("<td>", '<td style="text-align: left; padding: 4px; font-size: 12px;">')
            )
            st.markdown(html_detalle, unsafe_allow_html=True)

    # --------------------------------------------------------
    # 7. Gráfico de Control de Retornos
    # --------------------------------------------------------
    st.subheader("📊 Gráfico de Control de Retornos")
    fig, ax = plt.subplots(figsize=(8, 4))
    fechas = df_shewhart.index
    retornos = df_shewhart["Retorno"]

    ax.plot(fechas, retornos, marker="o", linestyle="-", label="Retorno", color="blue")
    ax.axhline(media, color="orange", linestyle="--", label="Media")

    # Límites según checkboxes
    if mostrar_1sigma:
        ax.axhline(media + std, color="green", linestyle="--", linewidth=0.8, label="+1σ")
        ax.axhline(media - std, color="green", linestyle="--", linewidth=0.8, label="-1σ")
    if mostrar_2sigma:
        ax.axhline(media + 2 * std, color="purple", linestyle="--", linewidth=0.8, label="+2σ")
        ax.axhline(media - 2 * std, color="purple", linestyle="--", linewidth=0.8, label="-2σ")
    if mostrar_3sigma:
        ax.axhline(media + 3 * std, color="red", linestyle="--", linewidth=1.0, label="+3σ")
        ax.axhline(media - 3 * std, color="red", linestyle="--", linewidth=1.0, label="-3σ")

    # Puntos violados (rojos)
    viols = []
    for _, row in df_detalle_viol.iterrows():
        viols.append(pd.to_datetime(row["Fecha"]))
    viol_indices = [fechas.get_loc(f) for f in viols if f in fechas]
    if viol_indices:
        ax.scatter(
            fechas[viol_indices],
            retornos.iloc[viol_indices],
            color="red",
            edgecolor="black",
            label="Violación",
            zorder=5
        )

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize="small")
    ax.set_title(f"Gráfico de Control de Retornos – {ticker}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # --------------------------------------------------------
    # 8. Botón para Descargar Excel con tres pestañas
    # --------------------------------------------------------
    st.subheader("📥 Exportar resultados")
    excel_buffer = generar_excel(
        df_precios=df_precios,
        df_shewhart=df_shewhart,
        df_tabla_viol=df_tabla_viol,
        nombre_empresa=nombre_empresa,
        ticker=ticker,
        intervalo=intervalo,
        periodos=periodos,
        tipo_retorno=tipo,
        media=media,
        std=std,
        dias_fuera_control=dias_fuera_control
    )
    st.download_button(
        label="📊 Descargar Excel (.xlsx)",
        data=excel_buffer,
        file_name=f"{ticker}_shewhart.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"❌ Error: {e}")
