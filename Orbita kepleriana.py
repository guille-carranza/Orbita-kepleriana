"""
Órbita kepleriana  r(θ) = p / (1 + e·cos θ)   (p > 0, e >= 0)

Grupo G3 (base: la Tierra). Unidades: UA (unidades astronómicas).

Archivos de entrada (en la misma carpeta que este script):
  - datos_grupos_proyecto_kepler.csv : parámetros de todos los grupos (se filtra grupo == "G3")
        columnas usadas: grupo, e, p_AU, a_AU, alpha_2D_deg, h_AU, k_AU,
                         perihelio_q_AU, afelio_Q_AU
  - G3_observaciones_xy.csv          : 48 observaciones (x_AU, y_AU) con ruido observacional

Librerías:
  - numpy      : cálculo vectorial (cos, sin, linspace). Evita bucles for y es el estándar
                 para evaluar r(θ) sobre muchos ángulos de una sola vez.
  - pandas     : lectura de los .csv y filtrado por grupo (df[df["grupo"] == "G3"]).
  - matplotlib : visualizaciones (es la librería base de gráficos en Python).
  - pathlib    : (biblioteca estándar) arma rutas al CSV sin depender de desde dónde se ejecute.

Las funciones están pensadas para convivir en un mismo archivo: las siguientes
(polar, potencial, etc.) se agregan debajo y se llaman desde main().
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
CARPETA = Path(__file__).resolve().parent
ARCHIVO_PARAMETROS = CARPETA / "datos_grupos_proyecto_kepler.csv"
ARCHIVO_OBSERVACIONES = CARPETA / "G3_observaciones_xy.csv"
GRUPO = "G3"


# ----------------------------------------------------------------------------
# Lectura de datos
# ----------------------------------------------------------------------------
def _leer_csv(ruta):
    """Lee un CSV detectando el separador (',' o ';') y convirtiendo comas decimales."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Si algún número vino como texto con coma decimal ("1,5"), se convierte a float.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            convertido = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False),
                                       errors="coerce")
            if convertido.notna().all():
                df[col] = convertido
    return df


def cargar_parametros(ruta=ARCHIVO_PARAMETROS, grupo=GRUPO):
    """Devuelve un diccionario con los parámetros orbitales del grupo indicado."""
    df = _leer_csv(ruta)
    filtro = df["grupo"].astype(str).str.strip().str.upper() == grupo.upper()
    df_g = df[filtro]

    if df_g.empty:
        raise ValueError(f'No hay filas con grupo == "{grupo}" en {ruta.name}')
    if len(df_g) > 1:
        print(f"Aviso: hay {len(df_g)} filas para {grupo}; se usa la primera.")

    fila = df_g.iloc[0]
    return {
        "e": float(fila["e"]),
        "p": float(fila["p_AU"]),
        "a": float(fila["a_AU"]),
        "alpha_deg": float(fila["alpha_2D_deg"]),
        "h": float(fila["h_AU"]),
        "k": float(fila["k_AU"]),
        "q": float(fila["perihelio_q_AU"]),   # distancia al foco en el perihelio (θ = 0)
        "Q": float(fila["afelio_Q_AU"]),      # distancia al foco en el afelio (θ = π)
    }


def cargar_observaciones(ruta=ARCHIVO_OBSERVACIONES):
    """Devuelve (x_obs, y_obs) como arreglos de numpy (48 puntos esperados)."""
    df = _leer_csv(ruta)
    x_obs = df["x_AU"].to_numpy(dtype=float)
    y_obs = df["y_AU"].to_numpy(dtype=float)

    if len(x_obs) != 48:
        print(f"Aviso: se esperaban 48 observaciones y hay {len(x_obs)}.")
    return x_obs, y_obs


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def radio_orbita(theta, p, e):
    """r(θ) = p / (1 + e cos θ)."""
    return p / (1.0 + e * np.cos(theta))


def rango_theta(e, n=2000):
    """
    Ángulos θ para dibujar la órbita.
    - e < 1 (elipse/círculo): vuelta completa [0, 2π].
    - e >= 1 (parábola/hipérbola): r no está definido/diverge donde 1 + e cos θ <= 0,
      así que se restringe a |θ| < arccos(-1/e) (con un margen para no dibujar el infinito).
    """
    if e < 1.0:
        return np.linspace(0.0, 2.0 * np.pi, n)
    theta_max = np.arccos(-1.0 / e)
    return np.linspace(-0.95 * theta_max, 0.95 * theta_max, n)


def orbita_cartesiana(p, e, alpha_deg, h, k, n=2000):
    """
    Calcula la órbita en el plano (x, y):
      1) sistema propio (foco en el origen):  x' = r cos θ,  y' = r sin θ
      2) rotación en alpha:                   x'' = x' cos α − y' sin α,  y'' = x' sin α + y' cos α
      3) traslación en (h, k):                x = x'' + h,  y = y'' + k
    """
    theta = rango_theta(e, n)
    r = radio_orbita(theta, p, e)
    xp, yp = r * np.cos(theta), r * np.sin(theta)

    alpha = np.deg2rad(alpha_deg)
    x = h + xp * np.cos(alpha) - yp * np.sin(alpha)
    y = k + xp * np.sin(alpha) + yp * np.cos(alpha)
    return x, y


def punto_en_orbita(distancia, theta, alpha_deg, h, k):
    """
    Posición (x, y) de un punto a 'distancia' del foco, en el ángulo θ del sistema propio.
    Perihelio: θ = 0, distancia = q.  Afelio: θ = π, distancia = Q.
    """
    alpha = np.deg2rad(alpha_deg)
    xp, yp = distancia * np.cos(theta), distancia * np.sin(theta)
    x = h + xp * np.cos(alpha) - yp * np.sin(alpha)
    y = k + xp * np.sin(alpha) + yp * np.cos(alpha)
    return x, y


# ----------------------------------------------------------------------------
# Función 1: órbita en el plano cartesiano
# ----------------------------------------------------------------------------
def graficar_orbita_cartesiana(parametros, x_obs, y_obs, guardar_como=None):
    """Órbita teórica + observaciones + foco (h, k) + perihelio, con ejes de igual escala."""
    p, e, a = parametros["p"], parametros["e"], parametros["a"]
    alpha, h, k = parametros["alpha_deg"], parametros["h"], parametros["k"]
    q, Q = parametros["q"], parametros["Q"]

    # Chequeo de coherencia: para una elipse debería cumplirse p = a (1 − e²)
    if e < 1.0:
        p_esperado = a * (1.0 - e**2)
        if not np.isclose(p, p_esperado, rtol=1e-2):
            print(f"Aviso: p_AU = {p:.4f} y a(1−e²) = {p_esperado:.4f} no coinciden.")
        if not np.isclose(q, a * (1.0 - e), rtol=1e-2):
            print(f"Aviso: q = {q:.4f} y a(1−e) = {a * (1.0 - e):.4f} no coinciden.")
        if not np.isclose(Q, a * (1.0 + e), rtol=1e-2):
            print(f"Aviso: Q = {Q:.4f} y a(1+e) = {a * (1.0 + e):.4f} no coinciden.")

    x_orb, y_orb = orbita_cartesiana(p, e, alpha, h, k)
    x_per, y_per = punto_en_orbita(q, 0.0, alpha, h, k)      # perihelio: θ = 0
    x_afe, y_afe = punto_en_orbita(Q, np.pi, alpha, h, k)    # afelio:    θ = π

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_orb, y_orb, color="tab:blue", lw=2,
            label=f"Órbita teórica  (e={e:.4f}, p={p:.4f} UA)")
    ax.scatter(x_obs, y_obs, s=30, color="tab:red", edgecolor="k", zorder=3,
               label=f"Observaciones ({len(x_obs)})")
    ax.scatter([h], [k], marker="o", s=40, color="gold", edgecolor="k", zorder=4,
               label=f"Foco (h, k) = ({h:.3f}, {k:.3f})")
    ax.scatter([x_per], [y_per], marker="D", s=60, color="tab:green", edgecolor="k", zorder=4,
               label=f"Perihelio q = {q:.4f} UA")
    if e < 1.0:   # el afelio solo existe en órbitas cerradas
        ax.scatter([x_afe], [y_afe], marker="s", s=60, color="tab:purple", edgecolor="k",
                   zorder=4, label=f"Afelio Q = {Q:.4f} UA")

    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")   # 1 UA en x = 1 UA en y (no deforma la elipse)
    ax.set_xlabel("x [UA]")
    ax.set_ylabel("y [UA]")
    ax.set_title(f"Órbita kepleriana en el plano cartesiano — {GRUPO}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")   # esquina inferior izquierda, lejos del foco

    fig.tight_layout()
    if guardar_como:
        fig.savefig(guardar_como, dpi=200)
    return fig, ax


# ----------------------------------------------------------------------------
# Función 2: órbita en forma polar
# ----------------------------------------------------------------------------
def observaciones_a_polares(x_obs, y_obs, parametros):
    """
    Pasa las observaciones (x, y) a coordenadas polares (θ, r) respecto del foco.
    Es la operación inversa a orbita_cartesiana():
      1) se resta la traslación (h, k)
      2) se deshace la rotación (se rota en −α)
      3) r = sqrt(x'² + y'²),  θ = atan2(y', x')   (θ en [0, 2π))
    """
    alpha = np.deg2rad(parametros["alpha_deg"])
    dx = x_obs - parametros["h"]
    dy = y_obs - parametros["k"]
 
    xp = dx * np.cos(alpha) + dy * np.sin(alpha)
    yp = -dx * np.sin(alpha) + dy * np.cos(alpha)
 
    r_obs = np.hypot(xp, yp)
    theta_obs = np.mod(np.arctan2(yp, xp), 2.0 * np.pi)
    return theta_obs, r_obs
 
 
def graficar_orbita_polar(parametros, x_obs, y_obs, theta_r_deg=60.0, guardar_como=None):
    """
    Órbita r(θ) en coordenadas polares (foco en el origen) + observaciones + perihelio/afelio.
    theta_r_deg: ángulo (en grados) en el que se dibuja el vector radio r(θ) como línea punteada.
    """
    p, e = parametros["p"], parametros["e"]
    q, Q = parametros["q"], parametros["Q"]
 
    theta = rango_theta(e)
    r = radio_orbita(theta, p, e)
    theta_obs, r_obs = observaciones_a_polares(x_obs, y_obs, parametros)
 
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
 
    ax.plot(theta, r, color="tab:blue", lw=2,
            label=f"Órbita teórica  (e={e:.4f}, p={p:.4f} UA)")
    ax.scatter(theta_obs, r_obs, s=30, color="tab:red", edgecolor="k", zorder=3,
               label=f"Observaciones ({len(x_obs)})")
    ax.scatter([0.0], [0.0], marker="o", s=40, color="gold", edgecolor="k", zorder=4,
               label="Foco (origen)")
    ax.scatter([0.0], [q], marker="D", s=60, color="tab:green", edgecolor="k", zorder=4,
               label=f"Perihelio q = {q:.4f} UA")
    if e < 1.0:   # el afelio solo existe en órbitas cerradas
        ax.scatter([np.pi], [Q], marker="s", s=60, color="tab:purple", edgecolor="k",
                   zorder=4, label=f"Afelio Q = {Q:.4f} UA")
 
    # Vector radio r(θ₀): línea punteada tenue desde el foco hasta la órbita
    theta_r = np.deg2rad(theta_r_deg)
    if 1.0 + e * np.cos(theta_r) <= 0.0:   # ángulo fuera del dominio (órbitas abiertas)
        theta_r = 0.0
    r_r = radio_orbita(theta_r, p, e)
    ax.plot([theta_r, theta_r], [0.0, r_r], linestyle="--", color="gray", lw=1.3,
            alpha=0.7, zorder=2, label=f"Radio r(θ={np.rad2deg(theta_r):.0f}°) = {r_r:.4f} UA")
    ax.annotate("r", xy=(theta_r, r_r / 2.0), xytext=(6, -2), textcoords="offset points",
                color="dimgray", fontsize=14, style="italic")
 
    ax.set_rlabel_position(135)          # etiquetas de r lejos de la órbita
    ax.set_title(f"Órbita kepleriana en forma polar — {GRUPO}\nr [UA], θ [°]", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=9)   # esquina inferior izquierda
 
    fig.tight_layout()
    if guardar_como:
        fig.savefig(guardar_como, dpi=200)
    return fig, ax
 

# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    parametros = cargar_parametros() #arreglo de parametros
    x_obs, y_obs = cargar_observaciones() #arreglo de observaciones
    print(f"Parámetros de {GRUPO}:")
    for clave, valor in parametros.items():
        print(f"  {clave:10s} = {valor}")
    while True:
        print("Seleccionar opcion: ")
        print("1. Graficar órbita cartesiana")
        print("2. Graficar órbita polar")
        print("3. Visualización de focos, ejes y sistema rotado")
        print("4. Visualizar superficie tridimensional del potencial gravitatorio")
        print("5. Visualizar curvas de nivel del potencial gravitatorio")
        print("6. Graficar campo vectorial asociado al campo gravitatorio")
        print("Otra opción para salir")
        opcion = input("Ingrese el número de la opción deseada: ").strip()
        if opcion == "1":
            graficar_orbita_cartesiana(parametros, x_obs, y_obs) #llamo a la funcion para graficar la orbita cartesiana
            plt.show()
        elif opcion == "2":
            graficar_orbita_polar(parametros, x_obs, y_obs) #llamo a la funcion para graficar la orbita polar
            plt.show()
        else:
            print("Programa cerrado.")
            break

if __name__ == "__main__":
    main()