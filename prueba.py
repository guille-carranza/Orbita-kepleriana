"""
TP - Órbita kepleriana (grupo G3)

Modelo:      r(θ) = p / (1 + e cos θ)           (foco en el origen del sistema orbital)
Potencial:   U(x, y) = -μ / sqrt(x² + y²)
Unidades:    UA para distancias, años para el tiempo  ->  μ = 4π² UA³/año²

Los CSV deben estar en la MISMA carpeta que este archivo.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares

# =====================================================================
# 0. CONFIGURACIÓN Y DATOS
# =====================================================================
GRUPO   = "G3"
GUARDAR = False     # True -> además de mostrar, guarda cada figura como PNG

CARPETA = os.path.dirname(os.path.abspath(__file__))

def ruta(nombre):
    """Ruta completa de un archivo que está junto a este script."""
    return os.path.join(CARPETA, nombre)

# --- Parámetros de la órbita de este grupo (una fila de la tabla) ---
grupos = pd.read_csv(ruta("datos_grupos_proyecto_kepler.csv"))
g = grupos[grupos["grupo"] == GRUPO].iloc[0]

E    = g["e"]                          # excentricidad
P    = g["p_AU"]                       # parámetro p
A    = g["a_AU"]                       # semieje mayor
ALFA = np.radians(g["alpha_2D_deg"])   # ángulo de rotación (en radianes)
H, K = g["h_AU"], g["k_AU"]            # traslación
MU   = 4*np.pi**2

R_MIN = P/(1 + E)                      # perihelio
R_MAX = P/(1 - E)                      # afelio
LIM   = 2.3*R_MAX                      # tamaño de las ventanas de los gráficos

# --- Observaciones ---
obs = pd.read_csv(ruta(f"{GRUPO}_observaciones_xy.csv"), encoding="utf-8-sig")
x_obs = obs["x_AU"].to_numpy()
y_obs = obs["y_AU"].to_numpy()


# =====================================================================
# 1. MODELO
# =====================================================================
def r_orbita(theta, p=P, e=E):
    """Radio en función del ángulo (coordenadas polares, foco en el origen)."""
    return p / (1 + e*np.cos(theta))

def orbita_xy(theta, p=P, e=E, alfa=ALFA, h=H, k=K):
    """Órbita en el plano cartesiano: se calcula en el sistema orbital,
    se rota un ángulo alfa y se traslada (h, k)."""
    r = r_orbita(theta, p, e)
    X, Y = r*np.cos(theta), r*np.sin(theta)         # sistema orbital
    x = h + X*np.cos(alfa) - Y*np.sin(alfa)         # rotar y trasladar
    y = k + X*np.sin(alfa) + Y*np.cos(alfa)
    return x, y

def semiejes(p, e):
    a = p/(1 - e**2)
    b = p/np.sqrt(1 - e**2)
    c = a*e
    return a, b, c

def U(x, y):
    """Potencial gravitatorio (con el foco en el origen del sistema que se le pase)."""
    with np.errstate(divide="ignore"):
        return -MU/np.sqrt(x**2 + y**2)

def grad_U(x, y):
    """Gradiente de U: (dU/dx, dU/dy)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (x**2 + y**2)**1.5
        return MU*x/d, MU*y/d

def finalizar(nombre):
    """Guarda la figura si GUARDAR está activado."""
    if GUARDAR:
        plt.savefig(ruta(nombre), dpi=200, bbox_inches="tight")


# =====================================================================
# 2. GRÁFICOS DE LA PARTE IV
# =====================================================================
def grafico_1_cartesiano():
    """Consigna 1: órbita en el plano cartesiano."""
    theta = np.linspace(0, 2*np.pi, 1000)
    x, y = orbita_xy(theta)
    xa, ya = orbita_xy(np.array([0.0, np.pi]))      # perihelio y afelio

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, label="Órbita")
    plt.plot(H, K, "ro", label="Foco")
    plt.plot(xa[0], ya[0], "g^", label="Perihelio")
    plt.plot(xa[1], ya[1], "bv", label="Afelio")
    plt.xlabel("x [UA]"); plt.ylabel("y [UA]")
    plt.title("Órbita en coordenadas cartesianas")
    plt.axis("equal"); plt.grid(True); plt.legend()
    finalizar("fig1_cartesiano.png")


def grafico_2_polar():
    """Consigna 2: la misma órbita en coordenadas polares (r en función de θ)."""
    theta = np.linspace(0, 2*np.pi, 1000)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(projection="polar")
    ax.plot(theta, r_orbita(theta), label="r(θ)")
    ax.plot(0, 0, "ro", label="Foco")
    ax.set_title("Órbita en coordenadas polares")
    ax.legend(loc="lower right")
    finalizar("fig2_polar.png")


def dibujar_ejes(p, e, alfa, h, k, titulo, archivo):
    """Foco, centro, ejes de la elipse y sistema rotado (x', y')."""
    a, b, c = semiejes(p, e)
    theta = np.linspace(0, 2*np.pi, 1000)
    x, y = orbita_xy(theta, p, e, alfa, h, k)

    u = np.array([np.cos(alfa), np.sin(alfa)])      # dirección del eje x' (hacia el perihelio)
    v = np.array([-np.sin(alfa), np.cos(alfa)])     # dirección del eje y'
    foco   = np.array([h, k])
    centro = foco - c*u                              # el centro queda del lado opuesto al perihelio
    mayor  = np.array([centro - a*u, centro + a*u])
    menor  = np.array([centro - b*v, centro + b*v])
    L = a                                            # largo de las flechas de x', y'

    plt.figure(figsize=(8, 7))
    plt.plot(x, y, label="Órbita")
    plt.plot(h, k, "ro", label="Foco")
    plt.plot(centro[0], centro[1], "ks", label="Centro de la elipse")
    plt.plot(mayor[:, 0], mayor[:, 1], "g--", label="Eje mayor")
    plt.plot(menor[:, 0], menor[:, 1], "m--", label="Eje menor")

    plt.axhline(0, color="gray", lw=0.6)             # ejes originales
    plt.axvline(0, color="gray", lw=0.6, label="Ejes originales (x, y)")

    for d, nombre, color in ((u, "x'", "orange"), (v, "y'", "purple")):
        punta = foco + L*d
        plt.plot(punta[0], punta[1], alpha=0)        # punto invisible: hace que la ventana incluya la flecha
        plt.annotate("", xy=punta, xytext=foco,
                     arrowprops=dict(arrowstyle="->", color=color, lw=2))
        plt.text(punta[0], punta[1], nombre, color=color, fontsize=12)

    plt.xlabel("x [UA]"); plt.ylabel("y [UA]")
    plt.title(titulo)
    plt.axis("equal"); plt.grid(True); plt.legend(fontsize=8)
    finalizar(archivo)


def grafico_3_ejes_y_rotado():
    """Consigna 3: foco, ejes y sistema rotado."""
    # (a) con los datos reales del grupo (casi una circunferencia)
    dibujar_ejes(P, E, ALFA, H, K,
                 "Foco, ejes y sistema rotado (datos del grupo)",
                 "fig3a_ejes.png")
    # (b) figura ILUSTRATIVA con excentricidad exagerada, para ver bien la geometría
    dibujar_ejes(3.2, 0.6, ALFA, 0.0, 0.0,
                 "ILUSTRATIVO: misma construcción con e = 0.6 (no son los datos del grupo)",
                 "fig3b_ejes_ilustrativo.png")


def grafico_4_superficie():
    """Consigna 4: superficie 3D de U(x, y), con el potencial centrado en el foco."""
    x = np.linspace(-LIM, LIM, 200)
    y = np.linspace(-LIM, LIM, 200)
    X, Y = np.meshgrid(x, y)
    Z = U(X - H, Y - K)
    Z = np.clip(Z, -MU/(0.4*R_MIN), 0)               # recorte (solo estético) cerca del foco

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("x [UA]"); ax.set_ylabel("y [UA]"); ax.set_zlabel("U(x,y)")
    ax.set_title("Superficie del potencial U(x,y)")
    finalizar("fig4_superficie.png")


def grafico_5_curvas_nivel():
    """Consigna 5: curvas de nivel de U, con la órbita superpuesta."""
    x = np.linspace(-LIM, LIM, 500)
    y = np.linspace(-LIM, LIM, 500)
    X, Y = np.meshgrid(x, y)
    Z = U(X - H, Y - K)

    radios  = [0.5*R_MIN, 0.75*R_MIN, R_MIN, R_MAX, 1.5*R_MAX, 2*R_MAX]
    niveles = sorted(-MU/r for r in radios)          # U = -μ/R  <->  circunferencia de radio R

    plt.figure(figsize=(7, 7))
    cs = plt.contour(X, Y, Z, levels=niveles)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    theta = np.linspace(0, 2*np.pi, 1000)
    xo, yo = orbita_xy(theta)
    plt.plot(xo, yo, "k", lw=2, label="Órbita")
    plt.plot(H, K, "ro", label="Foco")
    plt.xlabel("x [UA]"); plt.ylabel("y [UA]")
    plt.title("Curvas de nivel de U(x,y)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    finalizar("fig5_curvas_nivel.png")


def grafico_6_campo():
    """Consigna 6: campo vectorial -∇U (flechas normalizadas, color = módulo)."""
    pasos = 24                                       # cantidad PAR: la grilla no pisa el foco
    x = np.linspace(-LIM, LIM, pasos)
    y = np.linspace(-LIM, LIM, pasos)
    X, Y = np.meshgrid(x, y)

    gx, gy = grad_U(X - H, Y - K)
    Fx, Fy = -gx, -gy
    modulo = np.sqrt(Fx**2 + Fy**2)

    plt.figure(figsize=(7, 7))
    q = plt.quiver(X, Y, Fx/modulo, Fy/modulo, modulo, cmap="plasma", norm=LogNorm())
    plt.colorbar(q, label="|−∇U|  (escala logarítmica)")

    theta = np.linspace(0, 2*np.pi, 1000)
    xo, yo = orbita_xy(theta)
    plt.plot(xo, yo, "k", lw=2, label="Órbita")
    plt.plot(H, K, "ro", label="Foco")
    plt.xlabel("x [UA]"); plt.ylabel("y [UA]")
    plt.title("Campo vectorial −∇U(x,y)")
    plt.axis("equal"); plt.legend()
    finalizar("fig6_campo.png")


# =====================================================================
# 3. CONTRASTE CON LAS OBSERVACIONES
# =====================================================================
def distancia_a_curva(x, y, cx, cy):
    """Distancia de cada punto (x, y) al punto más cercano de la curva (cx, cy)."""
    dx = x[:, None] - cx[None, :]                    # matriz (n_obs x n_curva)
    dy = y[:, None] - cy[None, :]
    return np.sqrt(dx**2 + dy**2).min(axis=1)

def grafico_7_observaciones():
    """Modelo de la tabla vs observaciones."""
    theta = np.linspace(0, 2*np.pi, 2000)
    xm, ym = orbita_xy(theta)
    d = distancia_a_curva(x_obs, y_obs, xm, ym)
    print(f"[Modelo de la tabla] distancia media = {d.mean():.4f} UA,  "
          f"RMS = {np.sqrt((d**2).mean()):.4f} UA")

    plt.figure(figsize=(7, 7))
    plt.plot(xm, ym, label="Modelo (tabla)")
    plt.plot(x_obs, y_obs, "k.", label="Observaciones")
    plt.plot(H, K, "ro", label="Foco según la tabla")
    plt.xlabel("x [UA]"); plt.ylabel("y [UA]")
    plt.title("Observaciones vs modelo")
    plt.axis("equal"); plt.grid(True); plt.legend()
    finalizar("fig7_observaciones.png")

def ajustar_traslacion():
    """Busca la traslación (h, k) que mejor encaja (p, e y alfa quedan fijos)."""
    theta = np.linspace(0, 2*np.pi, 2000)

    def residuos(q):
        xm, ym = orbita_xy(theta, h=q[0], k=q[1])
        return distancia_a_curva(x_obs, y_obs, xm, ym)

    sol = least_squares(residuos, x0=[0.0, 0.0])
    h_aj, k_aj = sol.x
    rms = np.sqrt(np.mean(residuos(sol.x)**2))
    print(f"[Ajuste] h = {h_aj:.4f} UA,  k = {k_aj:.4f} UA,  RMS = {rms:.4f} UA")
    return h_aj, k_aj

def grafico_8_ajuste(h_aj, k_aj):
    """Modelo ajustado vs observaciones."""
    theta = np.linspace(0, 2*np.pi, 2000)
    x0, y0 = orbita_xy(theta)
    x1, y1 = orbita_xy(theta, h=h_aj, k=k_aj)

    plt.figure(figsize=(7, 7))
    plt.plot(x0, y0, "--", color="gray", label="Modelo de la tabla")
    plt.plot(x1, y1, label="Modelo con (h, k) ajustado")
    plt.plot(x_obs, y_obs, "k.", label="Observaciones")
    plt.plot(h_aj, k_aj, "ro", label="Foco ajustado")
    plt.xlabel("x [UA]"); plt.ylabel("y [UA]")
    plt.title("Ajuste de la traslación")
    plt.axis("equal"); plt.grid(True); plt.legend()
    finalizar("fig8_ajuste.png")


# =====================================================================
# 4. PROGRAMA PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print(f"Grupo {GRUPO}: e = {E:.6f}, p = {P:.6f} UA, alfa = {np.degrees(ALFA):.4f}°, "
          f"(h, k) = ({H}, {K})")

    grafico_1_cartesiano()
    grafico_2_polar()
    grafico_3_ejes_y_rotado()
    grafico_4_superficie()
    grafico_5_curvas_nivel()
    grafico_6_campo()

    grafico_7_observaciones()
    h_aj, k_aj = ajustar_traslacion()
    grafico_8_ajuste(h_aj, k_aj)

    plt.show()      # abre todas las ventanas juntas