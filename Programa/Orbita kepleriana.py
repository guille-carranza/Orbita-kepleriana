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
from matplotlib.patches import Arc
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
# Función 3: sistema original (x, y) y sistema rotado (x', y'), focos y ejes
# ----------------------------------------------------------------------------
def a_sistema_original(xp, yp, alpha_deg, h, k):
    """
    Del sistema rotado (x', y') al original (x, y):  primero se rota en α y luego se traslada a (h, k).
        x = h + x' cos α − y' sin α
        y = k + x' sin α + y' cos α
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
 

def error_radial_rms(parametros, x_obs, y_obs):
    """RMS de (r_obs − r_modelo(θ_obs)) en UA: sirve para chequear que el α del CSV encaja con los datos."""
    theta_obs, r_obs = observaciones_a_polares(x_obs, y_obs, parametros)
    r_modelo = radio_orbita(theta_obs, parametros["p"], parametros["e"])
    return float(np.sqrt(np.mean((r_obs - r_modelo) ** 2)))
 
 
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
        print("Ingrese cualquier otra opción para salir")
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
            
        else:
            print("Programa cerrado.")
            break

if __name__ == "__main__":
    main()