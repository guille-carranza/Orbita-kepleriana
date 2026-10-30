"""
Órbita kepleriana  r(θ) = p / (1 + e·cos θ)   (p > 0, e >= 0)

Grupo G3 (base: la Tierra). Unidades: UA (unidades astronómicas).

Archivos de entrada (en la misma carpeta que este script):
  - datos_grupos_proyecto_kepler.csv : parámetros de todos los grupos (se filtra grupo == "G3")
        columnas usadas: grupo, e, p_AU, a_AU, alpha_2D_deg, h_AU, k_AU,
                         perihelio_q_AU, afelio_Q_AU
  - G3_observaciones_xy.csv          : 48 observaciones (x_AU, y_AU) con ruido observacional
  - datos_orbitas_sistema_solar.csv  : órbitas de referencia (se usa la fila de la Tierra, solo para verificar)

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
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from matplotlib.colors import to_hex, LogNorm, LinearSegmentedColormap
# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
CARPETA = Path(__file__).resolve().parent
ARCHIVO_PARAMETROS = CARPETA / "datos_grupos_proyecto_kepler.csv"
ARCHIVO_OBSERVACIONES = CARPETA / "G3_observaciones_xy.csv"
ARCHIVO_SISTEMA_SOLAR = CARPETA / "datos_orbitas_sistema_solar.csv"
GRUPO = "G3"

# Parámetro gravitatorio μ = G·M del Sol, en UA³/año². Sale de la 3ª ley de Kepler con a = 1 UA y
# T = 1 año (μ = 4π² a³ / T²). Con él, U(x, y) queda en UA²/año². Cambialo si tu consigna da otro μ.
MU = 4.0 * np.pi**2

ESCALA_COLORES = "viridis"

# ----------------------------------------------------------------------------
# Lectura de datos
# ----------------------------------------------------------------------------
def _leer_csv(ruta):
    """Lee un CSV detectando el separador (',' o ';') y convirtiendo comas decimales."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    with open(ruta, encoding="utf-8-sig") as f:
        encabezado = f.readline()
    sep = max((",", ";", "\t"), key=encabezado.count)     # el separador más frecuente del encabezado
    df = pd.read_csv(ruta, sep=sep, encoding="utf-8-sig")
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


def verificar_con_tierra(parametros, ruta=ARCHIVO_SISTEMA_SOLAR, objeto="Tierra"):
    """
    Compara los parámetros de G3 con la fila de la Tierra de datos_orbitas_sistema_solar.csv
    y calcula μ = 4π²·a³/T² (3ª ley de Kepler). Solo imprime en consola; devuelve μ (o None).
    """
    try:
        df = _leer_csv(ruta)
    except FileNotFoundError as error:
        print(f"Aviso: {error}. Se omite la verificación con la Tierra.")
        return None

    fila = df[df["objeto"].astype(str).str.strip().str.lower() == objeto.lower()]
    if fila.empty:
        print(f'Aviso: no hay una fila "{objeto}" en {ruta.name}.')
        return None
    fila = fila.iloc[0]

    # (nombre en consola, clave en parametros, columna en el CSV del sistema solar)
    comparaciones = [
        ("a [UA]", "a", "a_AU"),
        ("e", "e", "e"),
        ("p [UA]", "p", "p_AU"),
        ("q [UA]", "q", "perihelio_q_AU"),
        ("Q [UA]", "Q", "afelio_Q_AU"),
        ("alpha [°]", "alpha_deg", "alpha_2D_deg"),
    ]
    print(f"\nVerificación de {GRUPO} contra {objeto} ({ruta.name}):")
    print(f"  {'parámetro':10s} {'G3':>12s} {objeto:>12s}   resultado")
    for nombre, clave, columna in comparaciones:
        g, t = parametros[clave], float(fila[columna])
        if clave == "alpha_deg":
            dif = (g - t + 180.0) % 360.0 - 180.0          # diferencia angular
            coincide = abs(dif) < 0.01
        else:
            coincide = abs(g - t) <= 1e-3 * max(abs(t), 1e-12)   # diferencia relativa < 0.1 %
        print(f"  {nombre:10s} {g:12.6f} {t:12.6f}   {'coincide' if coincide else 'DIFIERE'}")

    mu = 4.0 * np.pi**2 * float(fila["a_AU"]) ** 3 / float(fila["periodo_anos_estimado"]) ** 2
    print(f"  μ = 4π²a³/T² = {mu:.4f} UA³/año²   (constante MU del código: {MU:.4f})")
    return mu


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
    return a_sistema_original(xp, yp, alpha_deg, h, k)


def punto_en_orbita(distancia, theta, alpha_deg, h, k):
    """
    Posición (x, y) de un punto a 'distancia' del foco, en el ángulo θ del sistema propio.
    Perihelio: θ = 0, distancia = q.  Afelio: θ = π, distancia = Q.
    """
    xp, yp = distancia * np.cos(theta), distancia * np.sin(theta)
    return a_sistema_original(xp, yp, alpha_deg, h, k)


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
    xp, yp = a_sistema_rotado(x_obs, y_obs, parametros["alpha_deg"],
                              parametros["h"], parametros["k"])

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
# Función 3: sistema original (x, y) y sistema rotado (x', y'), focos y ejes
# ----------------------------------------------------------------------------
def error_radial_rms(parametros, x_obs, y_obs):
    """RMS de (r_obs − r_modelo(θ_obs)) en UA: sirve para chequear que el α del CSV encaja con los datos."""
    theta_obs, r_obs = observaciones_a_polares(x_obs, y_obs, parametros)
    r_modelo = radio_orbita(theta_obs, parametros["p"], parametros["e"])
    return float(np.sqrt(np.mean((r_obs - r_modelo) ** 2)))

def a_sistema_original(xp, yp, alpha_deg, h, k):
    """
    Del sistema rotado (x', y') al original (x, y):  primero se rota en α y luego se traslada a (h, k).
        x = h + x' cos α − y' sin α
        y = k + x' sin α + y' cos  α
    """
    alpha = np.deg2rad(alpha_deg)
    x = h + xp * np.cos(alpha) - yp * np.sin(alpha)
    y = k + xp * np.sin(alpha) + yp * np.cos(alpha)
    return x, y


def a_sistema_rotado(x, y, alpha_deg, h, k):
    """
    Del sistema original (x, y) al rotado (x', y') con origen en el foco (h, k):  es la inversa de
    a_sistema_original (se resta (h, k) y se rota en −α). El eje x' apunta al perihelio.
        x' =  (x − h) cos α + (y − k) sin α
        y' = −(x − h) sin α + (y − k) cos α
    """
    alpha = np.deg2rad(alpha_deg)
    dx, dy = x - h, y - k
    xp = dx * np.cos(alpha) + dy * np.sin(alpha)
    yp = -dx * np.sin(alpha) + dy * np.cos(alpha)
    return xp, yp


def graficar_sistema_rotado(parametros, x_obs, y_obs, guardar_como=None):
    """
    Visualización del foco, los ejes y el sistema rotado. Dibuja a la vez:
      - sistema original (x, y), con origen en (0, 0)       -> ejes grises punteados
      - sistema rotado (x', y'): MISMO origen (0, 0), solo rotado en α (sin traslación)
                                                            -> ejes negros con flechas
      - los focos de la elipse (F1 = (h, k) y F2, solo si e < 1)
      - los ejes de la elipse (mayor y menor), paralelos a x' e y'
      - el ángulo de rotación α (de x a x', antihorario) marcado con un arco
    El ángulo de rotación es α = alpha_2D_deg del CSV.
    """
    p, e, a = parametros["p"], parametros["e"], parametros["a"]
    alpha, h, k = parametros["alpha_deg"], parametros["h"], parametros["k"]
    q, Q = parametros["q"], parametros["Q"]

    x_orb, y_orb = orbita_cartesiana(p, e, alpha, h, k)
    x_per, y_per = punto_en_orbita(q, 0.0, alpha, h, k)
    x_afe, y_afe = punto_en_orbita(Q, np.pi, alpha, h, k)

    alpha_r = np.deg2rad(alpha)
    u = np.array([np.cos(alpha_r), np.sin(alpha_r)])      # dirección de x'
    v = np.array([-np.sin(alpha_r), np.cos(alpha_r)])     # dirección de y'
    origen = np.array([0.0, 0.0])

    # Largo de los ejes del sistema: algo mayor que la órbita
    L = 1.25 * (Q if e < 1.0 else 3.0 * q)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Órbita y datos
    ax.plot(x_orb, y_orb, color="tab:blue", lw=2,
            label=f"Órbita teórica  (e={e:.4f}, p={p:.4f} UA)")
    ax.scatter(x_obs, y_obs, s=30, color="tab:red", edgecolor="k", zorder=3,
               label=f"Observaciones ({len(x_obs)})")

    # Ejes de la elipse (solo órbitas cerradas): pasan por el centro, paralelos a x' e y'
    if e < 1.0:
        c = a * e
        b = a * np.sqrt(1.0 - e**2)
        xc, yc = a_sistema_original(-c, 0.0, alpha, h, k)           # centro de la elipse
        centro = np.array([xc, yc])
        ax.plot(*zip(centro - a * u, centro + a * u), color="tab:green", ls=":", lw=1.6,
                label="Ejes de la elipse (mayor y menor)")
        ax.plot(*zip(centro - b * v, centro + b * v), color="tab:green", ls=":", lw=1.6)

    # Sistema original (x, y): ejes grises punteados por el origen
    ax.plot([-L, L], [0, 0], color="gray", ls="--", lw=1.2, alpha=0.8,
            label="Sistema original (x, y)")
    ax.plot([0, 0], [-L, L], color="gray", ls="--", lw=1.2, alpha=0.8)
    for punta, texto in (((L, 0.0), "$x$"), ((0.0, L), "$y$")):
        ax.annotate("", xy=punta, xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.4))
        ax.annotate(texto, xy=punta, xytext=(6, 6), textcoords="offset points",
                    color="dimgray", fontsize=14)

    # Sistema rotado (x', y'): mismo origen, ejes negros
    ax.plot(*zip(origen - L * u, origen + L * u), color="black", lw=1.2, alpha=0.8,
            label="Sistema rotado (x′, y′)")
    ax.plot(*zip(origen - L * v, origen + L * v), color="black", lw=1.2, alpha=0.8)
    for punta, texto in ((origen + L * u, "$x'$"), (origen + L * v, "$y'$")):
        ax.annotate("", xy=tuple(punta), xytext=tuple(origen), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
        ax.annotate(texto, xy=tuple(punta), xytext=(6, 6), textcoords="offset points",
                    color="black", fontsize=14)

    # Ángulo α entre x y x' (arco centrado en el origen)
    alpha_disp = (alpha + 180.0) % 360.0 - 180.0          # llevado a (−180°, 180°]
    r_arco = 0.3 * L
    ax.add_patch(Arc((0.0, 0.0), 2 * r_arco, 2 * r_arco, angle=0.0,
                     theta1=min(0.0, alpha_disp), theta2=max(0.0, alpha_disp),
                     color="tab:orange", lw=2))
    medio = np.deg2rad(alpha_disp / 2.0)
    ax.annotate(f"α = {alpha:.1f}°",
                xy=(1.12 * r_arco * np.cos(medio), 1.12 * r_arco * np.sin(medio)),
                color="tab:orange", fontsize=12, ha="left", va="center", fontweight="bold")

    # Focos, perihelio y afelio
    ax.scatter([h], [k], marker="o", s=40, color="gold", edgecolor="k", zorder=6,
               label=f"Foco F₁ = (h, k) = ({h:.3f}, {k:.3f})")
    if e < 1.0:
        x_f2, y_f2 = a_sistema_original(-2.0 * a * e, 0.0, alpha, h, k)
        ax.scatter([x_f2], [y_f2], marker="o", s=40, color="tab:cyan", edgecolor="k", zorder=6,
                   label=f"Foco F₂ = ({x_f2:.3f}, {y_f2:.3f})")
        ax.scatter([x_afe], [y_afe], marker="s", s=60, color="tab:purple", edgecolor="k",
                   zorder=4, label=f"Afelio Q = {Q:.4f} UA")
    ax.scatter([x_per], [y_per], marker="D", s=60, color="tab:green", edgecolor="k", zorder=4,
               label=f"Perihelio q = {q:.4f} UA")

    # Límites que incluyan órbita y ejes
    xs = np.concatenate([x_orb, [-L, L, -L * u[0], L * u[0], -L * v[0], L * v[0]]])
    ys = np.concatenate([y_orb, [-L, L, -L * u[1], L * u[1], -L * v[1], L * v[1]]])
    margen = 0.08 * max(np.ptp(xs), np.ptp(ys))
    ax.set_xlim(xs.min() - margen, xs.max() + margen)
    ax.set_ylim(ys.min() - margen, ys.max() + margen)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [UA]")
    ax.set_ylabel("y [UA]")
    ax.set_title(f"Sistema original y sistema rotado en α (mismo origen) — {GRUPO}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    if guardar_como:
        fig.savefig(guardar_como, dpi=200)
    return fig, ax

 #----------------------------------------------------------------------------
# Centro del potencial (foco) y funciones 4 y 5: potencial gravitatorio
# ----------------------------------------------------------------------------
def estimar_foco_observaciones(parametros, x_obs, y_obs, iteraciones=100, tol=1e-12):
    """
    Estima el foco "real" (fx, fy) a partir de las 48 observaciones.
    Se conservan α, p y e de la tabla y solo se ajusta la posición del foco F por mínimos cuadrados
    (Gauss-Newton). Sale de la ecuación de la órbita:  r + e·x' = p,  con r = |P − F| y x' la
    componente de (P − F) a lo largo de x' (dirección del perihelio, ángulo α).
    Devuelve (fx, fy, rms): el foco y el error radial RMS de las observaciones respecto de ese foco.
    """
    e, p = parametros["e"], parametros["p"]
    alpha = np.deg2rad(parametros["alpha_deg"])
    u = np.array([np.cos(alpha), np.sin(alpha)])               # dirección de x'
    P = np.column_stack([x_obs, y_obs])
    F = P.mean(axis=0)                                         # punto de partida: el centro de los datos
 
    for _ in range(iteraciones):
        d = P - F
        dist = np.hypot(d[:, 0], d[:, 1])
        residuo = dist + e * (d @ u) - p                       # = 0 si P está sobre la órbita
        jacobiano = -(d / dist[:, None] + e * u)               # derivada del residuo respecto de F
        paso = np.linalg.lstsq(jacobiano, -residuo, rcond=None)[0]
        F = F + paso
        if np.linalg.norm(paso) < tol:
            break
 
    ajustado = dict(parametros, h=float(F[0]), k=float(F[1]))
    return float(F[0]), float(F[1]), error_radial_rms(ajustado, x_obs, y_obs)
 
 
def _centro_potencial(parametros, x_obs, y_obs, centro):
    """
    Devuelve (cx, cy): el centro del potencial, que físicamente es el foco (donde está el cuerpo
    que atrae). centro="observaciones" -> foco estimado con las observaciones (por defecto);
    centro="tabla" -> el foco (h, k) del CSV de grupos.
    """
    h, k = parametros["h"], parametros["k"]
    if centro == "observaciones":
        cx, cy, rms = estimar_foco_observaciones(parametros, x_obs, y_obs)
        print(f"\nCentro del potencial = foco estimado con las observaciones: ({cx:.4f}, {cy:.4f}) UA "
              f"(error radial RMS = {rms:.5f} UA)")
        print(f"  Foco de la tabla (h, k) = ({h:.4f}, {k:.4f}) UA; distancia entre ambos: "
              f"{np.hypot(cx - h, cy - k):.4f} UA")
    elif centro == "tabla":
        cx, cy = h, k
        print(f"\nCentro del potencial = foco de la tabla (h, k) = ({cx:.4f}, {cy:.4f}) UA")
    else:
        raise ValueError('centro debe ser "observaciones" o "tabla"')
    return cx, cy
 
 
def potencial_gravitatorio(x, y, mu=MU, centro=(0.0, 0.0)):
    """
    U = −μ / r, con r la distancia al 'centro' (el cuerpo atractor, ubicado en el foco):
    r = sqrt((x − cx)² + (y − cy)²). Con centro = (0, 0) es la fórmula U = −μ/sqrt(x² + y²).
    Funciona con números y con arreglos de numpy (μ > 0).
    """
    with np.errstate(divide="ignore"):          # en el centro devuelve −inf en lugar de fallar
        return -mu / np.hypot(x - centro[0], y - centro[1])
 
 
# ----------------------------------------------------------------------------
# Función 4: superficie tridimensional del potencial gravitatorio
# ----------------------------------------------------------------------------
def graficar_superficie_potencial(parametros, x_obs, y_obs, mu=MU, n=200,
                                  centro="observaciones", guardar_como=None):
    """
    Superficie z = U(x, y) con el potencial centrado en el foco (ver _centro_potencial) y las
    observaciones "subidas" a la superficie (z = U(x_obs, y_obs)), unidas en el orden de observación.
    """
    if mu <= 0:
        raise ValueError("μ debe ser positivo")
 
    e, q, Q = parametros["e"], parametros["q"], parametros["Q"]
    cx, cy = _centro_potencial(parametros, x_obs, y_obs, centro)
 
    # 1) Dominio: cuadrado de lado 2R centrado en el foco, algo mayor que la órbita
    R = 1.3 * (Q if e < 1.0 else 3.0 * q)
    X, Y = np.meshgrid(cx + np.linspace(-R, R, n), cy + np.linspace(-R, R, n))
 
    # 2) Potencial en la grilla; se recorta un círculo chico en el centro, donde U → −∞
    r_corte = 0.08 * R
    U = potencial_gravitatorio(X, Y, mu, (cx, cy))
    U = np.where(np.hypot(X - cx, Y - cy) < r_corte, np.nan, U)
    z_min = -mu / r_corte
 
    # 3) Observaciones sobre la superficie (solo las que caen dentro del dominio)
    dentro = ((np.abs(x_obs - cx) <= R) & (np.abs(y_obs - cy) <= R)
              & (np.hypot(x_obs - cx, y_obs - cy) >= r_corte))
    xo, yo = x_obs[dentro], y_obs[dentro]
    zo = potencial_gravitatorio(xo, yo, mu, (cx, cy)) + 0.01 * abs(z_min)   # leve elevación: no queda tapada
 
    # 4) Gráfico
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)   # respeta zorder
    superficie = ax.plot_surface(X, Y, U, cmap="viridis", alpha=0.85, linewidth=0,
                                 antialiased=True, rstride=2, cstride=2, zorder=1)
    if len(xo) > 0:
        # línea que une las observaciones en orden (se cierra con la primera) + los puntos observados
        ax.plot(np.append(xo, xo[0]), np.append(yo, yo[0]), np.append(zo, zo[0]),
                color="tab:red", lw=1.8, zorder=10)
        ax.scatter(xo, yo, zo, color="tab:red", edgecolor="k", s=18, depthshade=False, zorder=11,
                   label=f"Observaciones ({len(xo)})")
 
    ax.set_zlim(z_min, 0.0)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=28, azim=-60)
    ax.set_xlabel("x [UA]")
    ax.set_ylabel("y [UA]")
    ax.set_zlabel("U [UA²/año²]")
    ax.set_title(f"Potencial gravitatorio U = −μ/r centrado en el foco ({cx:.3f}, {cy:.3f}),  "
                 f"μ = {mu:.3f} — {GRUPO}", fontsize=10)
    fig.colorbar(superficie, ax=ax, shrink=0.6, pad=0.1, label="U [UA²/año²]")
    ax.legend(loc="lower left")
 
    fig.tight_layout()
    if guardar_como:
        fig.savefig(guardar_como, dpi=200)
    return fig, ax
 
 
# ----------------------------------------------------------------------------
# Función 5: curvas de nivel del potencial gravitatorio
# ----------------------------------------------------------------------------
def graficar_curvas_nivel_potencial(parametros, x_obs, y_obs, mu=MU, n=400, n_niveles=8,
                                    centro="observaciones", guardar_como=None):
    """
    Curvas de nivel U = k, con k < 0 (U = −μ/r siempre es negativo), con bandas de color sólido.
    El potencial está centrado en el foco, así que cada nivel es una circunferencia de radio
    r = −μ/k centrada en el foco. Cada banda entre dos niveles consecutivos se pinta con un color de
    ESCALA_COLORES (más oscuro = más profundo = U más negativo). Las observaciones se dibujan en rojo,
    y los dos niveles que pasan por la observación más cercana y más lejana al foco, en negro punteado.
    Imprime en consola cada constante k y el color de cada banda.
    """
    if mu <= 0:
        raise ValueError("μ debe ser positivo")
 
    h, k = parametros["h"], parametros["k"]
    e, q, Q = parametros["e"], parametros["q"], parametros["Q"]
    cx, cy = _centro_potencial(parametros, x_obs, y_obs, centro)
 
    # 1) Dominio y grilla (centrados en el foco)
    R = 1.3 * (Q if e < 1.0 else 3.0 * q)
    X, Y = np.meshgrid(cx + np.linspace(-R, R, n), cy + np.linspace(-R, R, n))
    U = potencial_gravitatorio(X, Y, mu, (cx, cy))
 
    # 2) Constantes k de las curvas de nivel: se eligen por radio (r = −μ/k) y se convierten a k
    radios = np.linspace(0.25 * R, R, n_niveles)
    niveles = np.sort(-mu / radios)                      # k < 0, en orden creciente (lo pide contour)
 
    # 3) Bandas de color: n_niveles + 1 bandas con colores equiespaciados de la escala elegida
    limites = np.concatenate([[U.min()], niveles, [U.max()]])
    colores = plt.get_cmap(ESCALA_COLORES)(np.linspace(0.0, 1.0, len(niveles) + 1))
 
    # 4) Observaciones y sus niveles extremos (distancias mínima y máxima al foco)
    dentro = (np.abs(x_obs - cx) <= R) & (np.abs(y_obs - cy) <= R)
    xo, yo = x_obs[dentro], y_obs[dentro]
    r_obs = np.hypot(xo - cx, yo - cy)
    k_extremos = np.sort(np.array([-mu / r_obs.min(), -mu / r_obs.max()]))
 
    print(f"\nCurvas de nivel U = k   (μ = {mu:.4f} UA³/año², U en UA²/año², k < 0)")
    print(f"Escala de colores: {ESCALA_COLORES} (oscuro = más profundo, claro = más cerca de U = 0)")
    for i, color in enumerate(colores):
        inferior = "−∞" if i == 0 else f"{niveles[i - 1]:.3f}"
        superior = "0" if i == len(niveles) else f"{niveles[i]:.3f}"
        print(f"  banda {i + 1}: {inferior:>9s} < U < {superior:>8s}   color {to_hex(color)}")
    for nivel in niveles:
        print(f"  k = {nivel:9.3f}   ->  r = −μ/k = {-mu / nivel:.4f} UA")
    # la observación más cercana al foco tiene el U más negativo (k menor); la más lejana, el k mayor
    print(f"Niveles de las observaciones: k = {k_extremos[0]:.3f} (r = {r_obs.min():.4f} UA, la más cercana al foco)"
          f" y k = {k_extremos[1]:.3f} (r = {r_obs.max():.4f} UA, la más lejana)")
 
    # 5) Gráfico
    fig, ax = plt.subplots(figsize=(9, 8))
    relleno = ax.contourf(X, Y, U, levels=limites, colors=colores)
    lineas = ax.contour(X, Y, U, levels=niveles, colors="black", linewidths=0.8, alpha=0.6)
    etiquetas = ax.clabel(lineas, fmt={nivel: f"k={nivel:.1f}" for nivel in niveles}, fontsize=8)
    for texto in etiquetas:
        texto.set_alpha(1.0)                  # el alpha de las líneas no debe aclarar el texto
        texto.set_color("black")
        texto.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
 
    extremos = ax.contour(X, Y, U, levels=k_extremos, colors="black", linestyles="--", linewidths=1.8)
    etiquetas = ax.clabel(extremos, fmt={nivel: f"k={nivel:.2f}" for nivel in k_extremos},
                          fontsize=8, colors="black")
    for texto in etiquetas:
        texto.set_bbox(dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))
 
    # Observaciones: puntos rojos unidos en el orden de observación (se cierra con la primera)
    ax.plot(np.append(xo, xo[0]), np.append(yo, yo[0]), color="tab:red", lw=1.8, zorder=6)
    ax.scatter(xo, yo, s=22, color="tab:red", edgecolor="k", zorder=7)
    ax.scatter([cx], [cy], marker="o", s=60, color="gold", edgecolor="k", zorder=8)
    if centro == "observaciones":
        ax.scatter([h], [k], marker="X", s=70, color="white", edgecolor="k", zorder=8)
 
    barra = fig.colorbar(relleno, ax=ax, ticks=niveles, format="%.1f", shrink=0.8, pad=0.03)
    barra.set_label(f"Profundidad: U = k [UA²/año²]  (escala {ESCALA_COLORES}: más oscuro = más profundo)")
 
    ax.set_aspect("equal")
    ax.set_xlabel("x [UA]")
    ax.set_ylabel("y [UA]")
    ax.set_title(f"Curvas de nivel del potencial U = k < 0 centrado en el foco,  μ = {mu:.3f} — {GRUPO}",
                 fontsize=11)
 
    leyenda = [
        Line2D([], [], color="black", lw=0.8, label="Curvas de nivel U = k (k < 0)"),
        Line2D([], [], color="black", lw=1.8, ls="--", label="Niveles extremos de las observaciones"),
        Line2D([], [], color="tab:red", lw=1.8, marker="o", mec="k", label=f"Observaciones ({len(xo)})"),
        Line2D([], [], color="gold", marker="o", mec="k", ls="",
               label=f"Centro del potencial = foco ({cx:.3f}, {cy:.3f})"),
    ]
    if centro == "observaciones":
        leyenda.append(Line2D([], [], color="white", marker="X", mec="k", ls="",
                              label=f"Foco de la tabla (h, k) = ({h:.3f}, {k:.3f})"))
    ax.legend(handles=leyenda, loc="lower left", fontsize=9, framealpha=0.9)
 
    fig.tight_layout()
    if guardar_como:
        fig.savefig(guardar_como, dpi=200)
    return fig, ax


#----------------------------------------------------------------------------
#Funcion para graficar el campo vectorial
#----------------------------------------------------------------------------

def campo_gravitatorio(x, y, mu=MU, centro=(0.0, 0.0)):
    """
    Campo gravitatorio F = −∇U para U = −μ/r, con r = sqrt((x − cx)² + (y − cy)²):
        ∇U = μ (x − cx, y − cy) / r³        ->        F = −μ (x − cx, y − cy) / r³
    Es radial, apunta hacia el centro (el foco) y su norma es |F| = μ / r².
    Devuelve las componentes (Fx, Fy). En el centro no está definido (r = 0).
    """
    dx, dy = x - centro[0], y - centro[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r3 = np.hypot(dx, dy) ** 3
        return -mu * dx / r3, -mu * dy / r3
 
def graficar_campo_vectorial(parametros, x_obs, y_obs, mu=MU, n_flechas=18,
                             centro="observaciones", guardar_como=None):
    """
    Dos paneles:
      - Izquierda: campo F = −∇U en una grilla de n_flechas × n_flechas puntos. Todas las flechas
        miden lo mismo (muestran solo la DIRECCIÓN); el COLOR indica la norma |F| (escala log).
      - Derecha: la norma |F| = μ/r² en función de la distancia r al foco, con el rango de r que
        recorren las observaciones.
    """
    if mu <= 0:
        raise ValueError("μ debe ser positivo")
 
    h, k = parametros["h"], parametros["k"]
    e, q, Q = parametros["e"], parametros["q"], parametros["Q"]
    cx, cy = _centro_potencial(parametros, x_obs, y_obs, centro)
 
    # 1) Dominio y grilla de flechas, centrados en el foco (n_flechas par: ningún punto cae en el foco)
    R = 1.3 * (Q if e < 1.0 else 3.0 * q)
    paso = 2.0 * R / n_flechas
    puntos = cx + (np.arange(n_flechas) + 0.5) * paso - R
    puntos_y = cy + (np.arange(n_flechas) + 0.5) * paso - R
    X, Y = np.meshgrid(puntos, puntos_y)
 
    # 2) Campo en la grilla: norma y dirección (vector unitario)
    Fx, Fy = campo_gravitatorio(X, Y, mu, (cx, cy))
    norma = np.hypot(Fx, Fy)
    ux, uy = Fx / norma, Fy / norma                       # solo la dirección: todas las flechas miden igual
 
    # 3) Observaciones: distancias al foco y norma del campo en ellas
    r_obs = np.hypot(x_obs - cx, y_obs - cy)
    F_obs = mu / r_obs**2
 
    print(f"\nCampo vectorial F = −∇U = −μ (x − cx, y − cy)/r³,  |F| = μ/r²   (μ = {mu:.4f} UA³/año²)")
    print(f"  Dirección: radial hacia el foco ({cx:.4f}, {cy:.4f}).  Norma en la grilla: "
          f"de {norma.min():.2f} (más lejos) a {norma.max():.2f} UA/año² (más cerca)")
    print(f"  Norma en las observaciones: de {F_obs.min():.3f} (r = {r_obs.max():.4f} UA) "
          f"a {F_obs.max():.3f} UA/año² (r = {r_obs.min():.4f} UA)")
 
    # 4) Escala de colores: viridis invertida y recortada (sin el amarillo más claro, que no se vería
    #    sobre fondo blanco): oscuro = norma grande (cerca del foco), claro = norma chica (lejos)
    cmap = LinearSegmentedColormap.from_list(
        "viridis_r_recortada", plt.get_cmap(ESCALA_COLORES + "_r")(np.linspace(0.15, 1.0, 256)))
    norm = LogNorm(vmin=norma.min(), vmax=norma.max())
 
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.8, 1]})
 
    # --- Panel izquierdo: campo de direcciones coloreado por la norma
    largo = 0.6 * paso                                    # largo de TODAS las flechas (en UA)
    flechas = ax.quiver(X, Y, ux, uy, norma, cmap=cmap, norm=norm, angles="xy",
                        scale_units="xy", scale=1.0 / largo, pivot="mid", width=0.004)
    ax.plot(np.append(x_obs, x_obs[0]), np.append(y_obs, y_obs[0]), color="tab:red", lw=1.5, zorder=5)
    ax.scatter(x_obs, y_obs, s=18, color="tab:red", edgecolor="k", zorder=6)
    ax.scatter([cx], [cy], marker="o", s=70, color="gold", edgecolor="k", zorder=8)
    if centro == "observaciones":
        ax.scatter([h], [k], marker="X", s=70, color="white", edgecolor="k", zorder=8)
 
    barra = fig.colorbar(flechas, ax=ax, shrink=0.8, pad=0.02)
    barra.set_label("|F| = μ/r²  [UA/año²]   (escala logarítmica; oscuro = intenso, claro = débil)")
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(cy - R, cy + R)
    ax.set_aspect("equal")
    ax.set_xlabel("x [UA]")
    ax.set_ylabel("y [UA]")
    ax.set_title("Campo F = −∇U: dirección (flechas) y norma (color)", fontsize=11)
    ax.grid(True, alpha=0.3)
    leyenda = [
        Line2D([], [], color="0.35", marker=r"$\rightarrow$", ls="", ms=12,
               label="Dirección de F (mismo largo en todas)"),
        Line2D([], [], color="tab:red", lw=1.5, marker="o", mec="k", label=f"Observaciones ({len(x_obs)})"),
        Line2D([], [], color="gold", marker="o", mec="k", ls="",
               label=f"Foco = centro del potencial ({cx:.3f}, {cy:.3f})"),
    ]
    if centro == "observaciones":
        leyenda.append(Line2D([], [], color="white", marker="X", mec="k", ls="",
                              label=f"Foco de la tabla (h, k) = ({h:.3f}, {k:.3f})"))
    ax.legend(handles=leyenda, loc="lower left", fontsize=8, framealpha=0.9)
 
    # --- Panel derecho: norma en función de la distancia al foco
    r = np.linspace(0.08 * R, np.sqrt(2.0) * R, 400)
    F_r = mu / r**2
    ax2.plot(r, F_r, color="0.5", lw=1, zorder=1)
    ax2.scatter(r, F_r, c=F_r, cmap=cmap, norm=norm, s=14, zorder=2)
    ax2.axvspan(r_obs.min(), r_obs.max(), color="tab:red", alpha=0.25,
                label=f"r de las observaciones ({r_obs.min():.3f} a {r_obs.max():.3f} UA)")
    ax2.scatter([r_obs.min(), r_obs.max()], [F_obs.max(), F_obs.min()], color="tab:red",
                edgecolor="k", zorder=3)
    ax2.annotate(f"{F_obs.max():.1f}", (r_obs.min(), F_obs.max()), xytext=(-34, 8),
                 textcoords="offset points", color="tab:red", fontsize=9)
    ax2.annotate(f"{F_obs.min():.1f}", (r_obs.max(), F_obs.min()), xytext=(8, 8),
                 textcoords="offset points", color="tab:red", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xlabel("r = distancia al foco [UA]")
    ax2.set_ylabel("|F| [UA/año²]")
    ax2.set_title("Norma del campo: |F| = μ/r²", fontsize=11)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
 
    fig.suptitle(f"Campo vectorial asociado a −∇U(x, y) — {GRUPO}", fontsize=13)
    fig.tight_layout()
    if guardar_como:
        fig.savefig(guardar_como, dpi=200)
    return fig, (ax, ax2)

# ----------------------------------------------------------------------------
# MAIN
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
        print("Ingrese cualquier otro carácter para salir")
        opcion = input("Opción deseada: ").strip()
        if opcion == "1":
            graficar_orbita_cartesiana(parametros, x_obs, y_obs) #llamo a la funcion para graficar la orbita cartesiana
            plt.show()
        elif opcion == "2":
            graficar_orbita_polar(parametros, x_obs, y_obs) #llamo a la funcion para graficar la orbita polar
            plt.show()
        elif opcion == "3":
            print(f"Ángulo de rotación α = {parametros['alpha_deg']:.4f}°")
            print(f"Error radial RMS de las observaciones: " f"{error_radial_rms(parametros, x_obs, y_obs):.5f} UA")
            graficar_sistema_rotado(parametros, x_obs, y_obs) #llamo a la funcion para graficar el sistema rotado
            plt.show()
        elif opcion == "4":
            print("Elija si quiere tener como centro el foco de la tabla o las observaciones (se debe tener en cuenta que el de la tabla es inconsistente con las observaciones realizadas):")
            cen=input("Ingrese 'tabla' o 'observaciones': ").strip().lower()
            if(cen != "tabla" and cen != "observaciones"):
                print("Opción inválida. Se usará 'observaciones' por defecto.")
                cen = "observaciones"
            graficar_superficie_potencial(parametros, x_obs, y_obs, centro=cen) #llamo a la funcion para graficar la superficie del potencial gravitatorio
            plt.show()
        elif opcion == "5":
            print("Elija si quiere tener como centro el foco de la tabla o las observaciones (se debe tener en cuenta que el de la tabla es inconsistente con las observaciones realizadas):")
            cen=input("Ingrese 'tabla' o 'observaciones': ").strip().lower()
            if(cen != "tabla" and cen != "observaciones"):
                print("Opción inválida. Se usará 'observaciones' por defecto.")
                cen = "observaciones"
            graficar_curvas_nivel_potencial(parametros, x_obs, y_obs, centro=cen) #llamo a la funcion para graficar las curvas de nivel del potencial gravitatorio
            plt.show()
        elif opcion == "6":
            print("Elija si quiere tener como centro el foco de la tabla o las observaciones (se debe tener en cuenta que el de la tabla es inconsistente con las observaciones realizadas):")
            cen=input("Ingrese 'tabla' o 'observaciones': ").strip().lower()
            if(cen != "tabla" and cen != "observaciones"):
                 print("Opción inválida. Se usará 'observaciones' por defecto.")
                 cen = "observaciones"
            graficar_campo_vectorial(parametros, x_obs, y_obs, centro=cen) #llamo a la funcion para graficar el campo vectorial asociado al campo gravitatorio
            plt.show()
        else:
            print("Programa cerrado.")
            break

if __name__ == "__main__":
    main()