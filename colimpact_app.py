# -*- coding: utf-8 -*-
"""
ColImpact (Column Impact Analysis Tool) - versao Streamlit
Adaptado do notebook ColImpact.ipynb (Jose Matheus de Castro Rodrigues)

Analise nao linear (geometrica e fisica) de pilares de concreto armado
sujeitos a carregamentos laterais, incluindo o efeito de impacto de
veiculos por Forcas Estaticas Equivalentes (FEE).
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve
import scipy.integrate as integrate

st.set_page_config(page_title="ColImpact - Analise de Impacto em Pilares", layout="wide")

# ============================================================
# 1. FUNCOES DE PROCESSAMENTO (traduzidas do notebook original)
# ============================================================

def deslocamento_lateral_engaste(nNos, DeltaZ, Nk, q, QT, MT, EIef, Pos_Fveic=None, Fveic=0):
    A = lil_matrix((nNos + 1, nNos + 1))
    B = np.zeros((nNos + 1, 1))

    tem_impacto = (Pos_Fveic is not None) and (Fveic != 0)
    if tem_impacto:
        No_impacto_fisico = int(round(Pos_Fveic / DeltaZ)) + 1
        No_impacto_matriz = No_impacto_fisico - 2

    for i in range(nNos + 1):
        if i == 0:
            A[i, 0] = (EIef[0]) + (-2 * Nk * (DeltaZ ** 2) + EIef[0] + 4 * EIef[1] + EIef[2])
            A[i, 1] = Nk * (DeltaZ ** 2) - 2 * EIef[1] - 2 * EIef[2]
            A[i, 2] = EIef[2]
            B[i] = q * DeltaZ ** 4
        elif i == 1:
            A[i, 0] = Nk * DeltaZ ** 2 - 2 * EIef[1] - 2 * EIef[2]
            A[i, 1] = -2 * Nk * DeltaZ ** 2 + EIef[1] + 4 * EIef[2] + EIef[3]
            A[i, 2] = Nk * DeltaZ ** 2 - 2 * EIef[2] - 2 * EIef[3]
            A[i, 3] = EIef[3]
            B[i] = q * DeltaZ ** 4
        elif i == nNos - 1:
            A[i, i] = -EIef[-2]
            A[i, i - 1] = 2 * EIef[-2]
            A[i, i - 2] = -EIef[-2]
            B[i] = (MT * DeltaZ ** 2)
        elif i == nNos:
            A[i, i] = -EIef[-1]
            A[i, i - 1] = -Nk * (DeltaZ ** 2) + 2 * EIef[-1]
            A[i, i - 2] = -EIef[-1] + EIef[-3]
            A[i, i - 3] = Nk * (DeltaZ ** 2) - 2 * EIef[-3]
            A[i, i - 4] = EIef[-3]
            B[i] = 2 * QT * DeltaZ ** 3
        else:
            A[i, i - 2] = EIef[i]
            A[i, i - 1] = Nk * (DeltaZ ** 2) - 2 * EIef[i] - 2 * EIef[i + 1]
            A[i, i] = -2 * Nk * (DeltaZ ** 2) + EIef[i] + 4 * EIef[i + 2] + EIef[i + 2]
            A[i, i + 1] = Nk * (DeltaZ ** 2) - 2 * EIef[i + 1] - 2 * EIef[i + 2]
            A[i, i + 2] = EIef[i + 2]

            if tem_impacto and i == No_impacto_matriz:
                B[i] = q * (DeltaZ ** 4) + Fveic * DeltaZ ** 3
            else:
                B[i] = q * (DeltaZ ** 4)

    A = A.tocsr()
    Resultado = spsolve(A, B)
    Resultado = Resultado[:-2]

    W = np.zeros((nNos, 1))
    W[1:nNos, 0] = Resultado
    return W


def deslocamento_lateral_biapoiados(nNos, DeltaZ, Nk, q, MB, MT, EIef, Pos_Fveic=None, Fveic=0):
    A = lil_matrix((nNos + 2, nNos + 2))
    B = np.zeros((nNos + 2, 1))

    tem_impacto = (Pos_Fveic is not None) and (Fveic != 0)
    if tem_impacto:
        No_impacto_fisico = int(round(Pos_Fveic / DeltaZ)) + 2
        No_impacto_matriz = No_impacto_fisico - 1

    for i in range(nNos + 2):
        if i == 0:
            A[i, 1] = -EIef[1]
            A[i, 2] = -EIef[1]
            B[i] = MB * DeltaZ ** 2
        elif i == 1:
            A[i, 0] = EIef[0]
            A[i, 1] = Nk * (DeltaZ ** 2) - 2 * EIef[0] - 2 * EIef[1]
            A[i, 2] = Nk * (DeltaZ ** 2) - 2 * EIef[1] - 2 * EIef[2]
            A[i, 3] = EIef[2]
            B[i] = q * DeltaZ ** 4
        elif i == 2:
            A[i, 1] = EIef[1]
            A[i, 2] = -2 * Nk * (DeltaZ ** 2) + EIef[1] + 4 * EIef[2] + EIef[3]
            A[i, 3] = Nk * (DeltaZ ** 2) - 2 * EIef[2] - 2 * EIef[3]
            A[i, 4] = EIef[3]
            B[i] = q * DeltaZ ** 4
        elif i == 3:
            A[i, 2] = Nk * (DeltaZ ** 2) - 2 * EIef[2] - 2 * EIef[3]
            A[i, 3] = -2 * Nk * (DeltaZ ** 2) + EIef[2] + 4 * EIef[3] + EIef[4]
            A[i, 4] = Nk * (DeltaZ ** 2) - 2 * EIef[3] - 2 * EIef[4]
            A[i, 5] = EIef[4]
            B[i] = q * DeltaZ ** 4
        elif i == nNos - 2:
            A[i, i + 1] = Nk * (DeltaZ ** 2) - 2 * EIef[-4] - 2 * EIef[-3]
            A[i, i] = -2 * Nk * (DeltaZ ** 2) + EIef[-5] + 4 * EIef[-4] + EIef[-3]
            A[i, i - 1] = Nk * (DeltaZ ** 2) - 2 * EIef[-5] - 2 * EIef[-4]
            A[i, i - 2] = EIef[-5]
            B[i] = q * DeltaZ ** 4
        elif i == nNos - 1:
            A[i, i + 1] = EIef[-2]
            A[i, i] = -2 * Nk * (DeltaZ ** 2) + EIef[-4] + 4 * EIef[-3] + EIef[-2]
            A[i, i - 1] = Nk * (DeltaZ ** 2) - 2 * EIef[-4] - 2 * EIef[-3]
            A[i, i - 2] = EIef[-4]
            B[i] = q * DeltaZ ** 4
        elif i == nNos:
            A[i, i + 1] = EIef[-1]
            A[i, i] = Nk * (DeltaZ ** 2) - 2 * EIef[-2] - 2 * EIef[-1]
            A[i, i - 1] = Nk * (DeltaZ ** 2) - 2 * EIef[-3] - 2 * EIef[-2]
            A[i, i - 2] = EIef[-3]
            B[i] = q * DeltaZ ** 4
        elif i == nNos + 1:
            A[i, i - 1] = -EIef[-2]
            A[i, i - 2] = -EIef[-2]
            B[i] = MT * DeltaZ ** 2
        else:
            A[i, i - 2] = EIef[i - 1]
            A[i, i - 1] = Nk * (DeltaZ ** 2) - 2 * EIef[i - 1] - 2 * EIef[i]
            A[i, i] = -2 * Nk * (DeltaZ ** 2) + EIef[i - 1] + 4 * EIef[i] + EIef[i + 1]
            A[i, i + 1] = Nk * (DeltaZ ** 2) - 2 * EIef[i] - 2 * EIef[i + 1]
            A[i, i + 2] = EIef[i + 1]

            if tem_impacto and i == No_impacto_matriz:
                B[i] = q * DeltaZ ** 4 + Fveic * DeltaZ ** 3
            else:
                B[i] = q * DeltaZ ** 4

    A = A.tocsr()
    Resultado = spsolve(A, B)
    Resultado = Resultado[2:-2].reshape(-1, 1)

    W = np.vstack([0, Resultado, 0])
    return W


def momento_fletor_engaste(nNos, DeltaZ, EIef, MT, W):
    M = np.zeros(nNos)
    for i in range(1, nNos - 1):
        M[i] = -EIef[i] * (W[i - 1, 0] - 2 * W[i, 0] + W[i + 1, 0]) / DeltaZ ** 2
    M[0] = -EIef[0] * (2 * W[1, 0]) / DeltaZ ** 2
    M[-1] = MT
    return M


def momento_fletor_biapoiado(nNos, DeltaZ, EIef, MB, MT, W):
    M = np.zeros(nNos)
    for i in range(1, nNos - 1):
        M[i] = -EIef[i] * (W[i - 1, 0] - 2 * W[i, 0] + W[i + 1, 0]) / DeltaZ ** 2
    M[0] = MB
    M[-1] = MT
    return M


def rigidez_flexao_constante(b, h, r, fck, E, coef):
    if r != 0:
        I = (np.pi * r ** 4) / 4
    else:
        I = (b * h ** 3) / 12

    if E != 0:
        EIef = coef * I * E
    else:
        Ei = 5600 * (fck / 1000) ** 0.5
        Ecs = (0.8 + 0.2 * ((fck / 1000) / 80)) * Ei
        EIef = coef * I * Ecs * 1000
    return EIef


def rigidez_secante(chi_values, MF_values, MF, Mr):
    from scipy.interpolate import interp1d
    interp_chi = interp1d(MF_values, chi_values, kind='quadratic', fill_value="extrapolate")
    MF_limited = np.minimum(MF, MF_values[-1])
    chi_interp = interp_chi(MF_limited)
    chi_fiss = interp_chi(Mr)
    EI_const = Mr / chi_fiss
    EI_values = np.where(MF > Mr, MF / chi_interp, EI_const)
    return EI_values


def eps_bar(chi, d, eps_m):
    return eps_m + chi * d


def eps_lim(d, x, h, eps0, epsu, k, ds):
    xlim = (epsu / (epsu + 0.01)) * ds[0]

    if isinstance(d, np.ndarray):
        epsilon = np.full_like(d, np.nan, dtype=float)
    else:
        epsilon = np.nan

    x_scalar = x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
    if np.isnan(x_scalar):
        return epsilon

    if 0 <= x_scalar < xlim:
        denom = ds[0] - x_scalar
        if np.isclose(denom, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = 0.01 * ((x_scalar - d) / denom)
    elif xlim <= x_scalar <= h:
        if np.isclose(x_scalar, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = epsu * ((x_scalar - d) / x_scalar)
    elif x_scalar > h:
        denom = x_scalar - k * h
        if np.isclose(denom, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = eps0 * ((x_scalar - d) / denom)
    return epsilon


def sigma_concreto(epsc, eps0, epsu, fcd, coef):
    sigma_c = np.zeros_like(epsc)
    fck_MPa = (fcd * 1.4) / 1000

    eta_c = 1.0 if fck_MPa <= 40 else (40 / fck_MPa) ** (1 / 3)
    n = 2.0 if fck_MPa <= 50 else 1.4 + 23.4 * ((90 - fck_MPa) / 100) ** 4

    mask_parabolic = (epsc > 0) & (epsc <= eps0)
    mask_rectangular = (epsc > eps0) & (epsc <= epsu)

    if coef == 1.1:
        sigma_c[mask_parabolic] = 1.1 * fcd * (1 - (1 - epsc[mask_parabolic] / eps0) ** n)
        sigma_c[mask_rectangular] = 1.1 * fcd
    else:
        sigma_c[mask_parabolic] = coef * eta_c * fcd * (1 - (1 - epsc[mask_parabolic] / eps0) ** n)
        sigma_c[mask_rectangular] = coef * eta_c * fcd
    return sigma_c


def sigma_aco(epss, fyd, Es):
    eps_yd = fyd / Es
    sigma_sd = np.zeros_like(epss)
    for i in range(len(epss)):
        if abs(epss[i]) <= eps_yd:
            sigma_sd[i] = Es * epss[i]
        else:
            sigma_sd[i] = np.sign(epss[i]) * fyd
    return sigma_sd


def equilibrio_forcas_e_momentos(x, b, h, ds, n, fcd, fyd, Es, eps0, epsu, k, N, M):
    dc = np.arange(0, h, 0.01)
    dc = np.append(dc, h)
    epsc = eps_lim(dc, x, h, eps0, epsu, k, ds)
    epss = eps_lim(ds, x, h, eps0, epsu, k, ds)
    sigc = sigma_concreto(epsc, eps0, epsu, fcd, 0.85)
    sigs = sigma_aco(epss, fyd, Es)
    Rcc = integrate.trapezoid(sigc * b, dc)
    Mc = integrate.trapezoid(sigc * dc * b, dc)
    EQ = (N - Rcc) * np.sum(n * sigs * ds) + (M - N * (h / 2) + Mc) * np.sum(n * sigs)
    return EQ


def area_aco_total(x, b, h, ds, n, fcd, fyd, Es, eps0, epsu, k, N, M):
    dc = np.arange(0, h, 0.01)
    dc = np.append(dc, h)
    epsc = eps_lim(dc, x, h, eps0, epsu, k, ds)
    epss = eps_lim(ds, x, h, eps0, epsu, k, ds)
    sigc = sigma_concreto(epsc, eps0, epsu, fcd, 0.85)
    sigs = sigma_aco(epss, fyd, Es)
    Mc = integrate.trapezoid(sigc * dc * b, dc)
    Ast = (np.sum(n) * (-M + N * (h / 2) - Mc)) / np.sum(n * sigs * ds)
    return Ast


BITOLAS_COMERCIAIS = np.array([
    [10.0, 0.0000785],
    [12.5, 0.000122],
    [16.0, 0.000201],
    [20.0, 0.000314],
    [25.0, 0.000491],
    [32.0, 0.000804],
])


def bitola_comercial(As_barra_cal):
    for bitola, area in BITOLAS_COMERCIAIS:
        if area >= As_barra_cal:
            return area, bitola
    return None, None


def equilibrio_forcas(b, h, dc, ds, As, fcd, fyd, Es, eps0, epsu, N, epsm, chi, coef):
    epsc = eps_bar(chi, dc, epsm)
    epss = eps_bar(chi, ds, epsm)
    sigc = sigma_concreto(epsc, eps0, epsu, fcd, coef)
    sigs = sigma_aco(epss, fyd, Es)
    Rcc = integrate.trapezoid(sigc * b, dc)
    Rs = np.sum(As * sigs)
    return N - Rcc - Rs


def momento_equilibrante(b, h, dc, ds, As, fcd, fyd, Es, eps0, epsu, N, epsm, chi, coef):
    epsc = eps_bar(chi, dc, epsm)
    epss = eps_bar(chi, ds, epsm)
    sigc = sigma_concreto(epsc, eps0, epsu, fcd, coef)
    sigs = sigma_aco(epss, fyd, Es)
    Mc = integrate.trapezoid(sigc * dc * b, dc)
    Ms = np.sum(As * sigs * ds)
    return Mc + Ms


def momento_fissuracao(fck, b, h, D, x_raiz, alpha=1.5):
    fct_m = 0.3 * (fck / 1000) ** (2 / 3) * 1000
    fct = 0.7 * fct_m
    if D != 0:
        Ic = (np.pi * (D / 2) ** 4) / 4
        yt = D - x_raiz
    else:
        Ic = (b * h ** 3) / 12
        yt = h - x_raiz
    Mr = alpha * fct * (Ic / yt)
    return Mr


def gerar_diagrama_mchi(b, dc, ds, As, fcd, fyd, Es, eps0, epsu, Nd, coef,
                         incremento_chi=1e-5, Nd_referencia=None, Mdbreak=None,
                         progress_cb=None, max_iter=200000):
    """Gera a curva Momento x Curvatura ate a ruptura (ou ate Mdbreak, se informado)."""
    chi_values = []
    M_values = []
    epsm_chute = 0.001
    chi = 0.0
    Nd_eff = Nd_referencia if Nd_referencia is not None else Nd

    it = 0
    while it < max_iter:
        it += 1
        epsm_raiz = fsolve(lambda epsm: equilibrio_forcas(
            b, None, dc, ds, As, fcd, fyd, Es, eps0, epsu, Nd_eff, epsm, chi, coef), epsm_chute)

        M = momento_equilibrante(b, None, dc, ds, As, fcd, fyd, Es, eps0, epsu,
                                  Nd_eff, epsm_raiz, chi, coef)

        deformacoes = eps_bar(chi, dc, epsm_raiz[0])

        if Mdbreak is None:
            if np.any(deformacoes > 0.0035) or np.any(deformacoes < -0.01):
                break
        else:
            if M > Mdbreak:
                break

        chi_values.append(chi)
        M_values.append(M)

        epsm_chute = epsm_raiz[0]
        chi += incremento_chi

        if progress_cb is not None and it % 200 == 0:
            progress_cb(min(it / max_iter, 0.99))

    return np.array(chi_values), np.array(M_values)


def iterar_nao_linear(nNos, DeltaZ, Nd, qd, Cbase, Ctopo, EIef0, Pos_Fveic, Fveic,
                       chi_values_11, M_values_11, Mr, engastado, gamma_f3=1.1,
                       tol=1e-4, max_iter=100):
    """Loop iterativo (NLG + NLF) - Cbase/Ctopo: (QTd,MTd) se engastado ou (MBd,MTd) se biapoiado."""
    logs = []

    if engastado:
        QTd, MTd = Cbase, Ctopo
        W_prev = deslocamento_lateral_engaste(nNos, DeltaZ, Nd / gamma_f3, qd, QTd / gamma_f3,
                                               MTd / gamma_f3, EIef0, Pos_Fveic, Fveic)
        M_prev = momento_fletor_engaste(nNos, DeltaZ, EIef0, MTd / gamma_f3, W_prev)
    else:
        MBd, MTd = Cbase, Ctopo
        W_prev = deslocamento_lateral_biapoiados(nNos, DeltaZ, Nd / gamma_f3, qd / gamma_f3,
                                                  MBd / gamma_f3, MTd / gamma_f3, EIef0,
                                                  Pos_Fveic, Fveic)
        M_prev = momento_fletor_biapoiado(nNos, DeltaZ, EIef0, MBd / gamma_f3, MTd / gamma_f3, W_prev)

    Mr_val = float(np.atleast_1d(Mr)[0])
    EIef_NL = rigidez_secante(chi_values_11, M_values_11, np.abs(M_prev), Mr_val)
    pad_width = (0, 1) if engastado else (1, 1)
    EIef_NL_corrigido = np.pad(EIef_NL, pad_width, mode='edge')
    EIef_NL_final = np.copy(EIef_NL)

    for it in range(1, max_iter + 1):
        if engastado:
            W_new = deslocamento_lateral_engaste(nNos, DeltaZ, Nd / gamma_f3, qd, QTd / gamma_f3,
                                                  MTd / gamma_f3, EIef_NL_corrigido, Pos_Fveic, Fveic)
            M_new = momento_fletor_engaste(nNos, DeltaZ, EIef_NL_final, MTd / gamma_f3, W_new)
        else:
            W_new = deslocamento_lateral_biapoiados(nNos, DeltaZ, Nd / gamma_f3, qd / gamma_f3,
                                                     MBd / gamma_f3, MTd / gamma_f3,
                                                     EIef_NL_corrigido, Pos_Fveic, Fveic)
            M_new = momento_fletor_biapoiado(nNos, DeltaZ, EIef_NL_final, MBd / gamma_f3,
                                              MTd / gamma_f3, W_new)

        EIef_NL_final = rigidez_secante(chi_values_11, M_values_11, np.absolute(M_new), Mr_val)

        diff_W = np.linalg.norm(W_new - W_prev) / np.linalg.norm(W_new)
        logs.append((it, diff_W))

        if diff_W < tol:
            return W_new, M_new, EIef_NL_final, logs, True

        W_prev = np.copy(W_new)
        EIef_NL_corrigido = np.pad(EIef_NL_final, (0, 1), mode='edge')

    return W_new, M_new, EIef_NL_final, logs, False


def tabela_fs(arr_momentos, m_resistente):
    fs_numericos = []
    for m in arr_momentos:
        if abs(m) < 0.001:
            fs_numericos.append(float('inf'))
        else:
            fs_numericos.append(abs(m_resistente / m))
    min_fs = min(fs_numericos)

    fs_display = []
    for val in fs_numericos:
        if val == float('inf'):
            fs_display.append("(OK!)")
        else:
            val_fmt = round(val, 2)
            fs_display.append(f"{val_fmt} <=" if val == min_fs else val_fmt)

    num_pontos = len(arr_momentos)
    labels_z = []
    for i in range(num_pontos):
        ratio = i / (num_pontos - 1) if num_pontos > 1 else 0.0
        if ratio == 0.0:
            labels_z.append("0 (Base)")
        elif ratio >= 0.999:
            labels_z.append("L (Topo)")
        else:
            labels_z.append(f"{ratio:.2f} L")

    df = pd.DataFrame({'z': labels_z, 'Msd (kN.m)': np.round(arr_momentos, 2), 'F.S.': fs_display})
    return df.iloc[::-1].reset_index(drop=True), min_fs


# ============================================================
# 2. FUNCOES DE PLOTAGEM
# ============================================================

def _plt_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6,
    })


def plot_momento_curvatura(chi85, M85, chi11, M11):
    _plt_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(chi85, M85, color='red', linestyle='-', linewidth=2, label='0.85 fcd')
    ax.plot(chi11, M11, color='red', linestyle='--', linewidth=2, label='1.1 fcd (Nd/1.1)')
    ax.set_xlabel('Curvatura ($\\chi$) [1/m]')
    ax.set_ylabel('Momento Fletor [kN.m]')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def plot_deslocamentos(series_list, L):
    _plt_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 0], [0, L], color='black', linewidth=1.2, zorder=1)
    styles = ['-', '--', '-.', ':']
    for i, s in enumerate(series_list):
        y = np.linspace(0, L, len(s['Z']))
        ax.plot(np.array(s['Z']).flatten(), y, linestyle=styles[i % 4], linewidth=1.6,
                label=s['label'], zorder=2)
    ax.set_xlabel('Deslocamento [m]')
    ax.set_ylabel('Posicao ao longo do pilar [m]')
    ax.legend(loc='best')
    ax.set_ylim(0, L)
    fig.tight_layout()
    return fig


def plot_momentos(series_list, L):
    _plt_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 0], [0, L], color='black', linewidth=1.2, zorder=1)
    styles = ['-', '--', '-.', ':']
    for i, s in enumerate(series_list):
        y = np.linspace(0, L, len(s['M']))
        ax.plot(np.array(s['M']).flatten(), y, linestyle=styles[i % 4], linewidth=1.6,
                label=s['label'], zorder=2)
    ax.set_xlabel('Momento Fletor [kN.m]')
    ax.set_ylabel('Posicao ao longo do pilar [m]')
    ax.legend(loc='best')
    ax.set_ylim(0, L)
    fig.tight_layout()
    return fig


def plot_fs(arr_momentos, m_resistente, L, limite_escala=35.0):
    _plt_style()
    num_pontos = len(arr_momentos)
    z_vals = np.linspace(0, L, num_pontos)
    with np.errstate(divide='ignore', invalid='ignore'):
        fs_vals = np.abs(m_resistente / arr_momentos)
    fs_plot = np.clip(fs_vals, 0, limite_escala)
    min_fs_idx = np.argmin(fs_vals)
    min_fs = fs_vals[min_fs_idx]
    z_critico = z_vals[min_fs_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(0, 1.0, color='#ffe6e6', alpha=1.0, zorder=0)
    ax.axvline(x=1.0, color='#d62728', linestyle='--', linewidth=1.5, zorder=1,
               label='Limite ($M_{sd}=M_{rd}$)')
    color_line = '#003366'
    ax.plot(fs_plot, z_vals, color=color_line, linewidth=1.6, zorder=3, label='FSI calculado')
    ax.fill_betweenx(z_vals, 0, fs_plot, color=color_line, alpha=0.1, zorder=2)
    ax.scatter([min_fs if min_fs <= limite_escala else limite_escala], [z_critico], color='white',
               edgecolors='red', s=60, linewidth=1.5, zorder=4)
    ax.annotate(f'{min_fs:.2f}', xy=(min(min_fs, limite_escala), z_critico),
                xytext=(min(min_fs, limite_escala) + 2.0, z_critico),
                arrowprops=dict(arrowstyle='-|>', color='black', linewidth=1.0),
                fontsize=11, fontweight='bold', color='red', va='center', ha='left')
    ax.set_xlabel('Fator de Seguranca de Impacto (FSI)')
    ax.set_ylabel('Posicao ao longo do pilar [m]')
    ax.set_xlim(0, limite_escala)
    ax.set_ylim(0, L)
    ax.legend(loc='best')
    fig.tight_layout()
    return fig, min_fs


# ============================================================
# 3. INTERFACE STREAMLIT
# ============================================================

st.title("ColImpact - Analise de Impacto em Pilares de Concreto Armado")
st.caption("Adaptado do notebook ColImpact.ipynb - analise nao linear (geometrica e fisica) de "
           "pilares sujeitos a forcas estaticas equivalentes de impacto de veiculo.")

with st.sidebar:
    st.header("1. Configuracao geral")
    tipo_secao = st.radio("Tipo de secao", ["Retangular", "Circular"], horizontal=True)
    contorno = st.radio("Condicao de contorno", ["Biapoiado", "Engastado"], horizontal=True)
    engastado = contorno == "Engastado"

    st.header("2. Geometria")
    L = st.number_input("Comprimento do pilar L (m)", value=2.8, min_value=0.1, step=0.1)
    if tipo_secao == "Retangular":
        b = st.number_input("Base b (m)", value=0.5, min_value=0.01, step=0.01)
        h = st.number_input("Altura h (m)", value=0.2, min_value=0.01, step=0.01)
        D = 0.0
    else:
        D = st.number_input("Diametro D (m)", value=0.4, min_value=0.01, step=0.01)
        h = D
        b = None

    st.header("3. Discretizacao")
    DeltaZ = st.number_input("Passo de discretizacao (DeltaZ, m)", value=0.10, min_value=0.001,
                              step=0.01, format="%.3f")

    st.header("4. Armadura")
    dLinha = st.number_input("Cobrimento ao centro da barra d' (m)", value=0.05, min_value=0.005,
                              step=0.005, format="%.4f")

    if tipo_secao == "Retangular":
        n_str = st.text_input("Numero de barras por camada (separado por virgula)", value="4,4")
        n = np.array([int(v.strip()) for v in n_str.split(",") if v.strip() != ""])
        nLinha = len(n)
        ds = np.linspace(-h / 2 + dLinha, h / 2 - dLinha, nLinha)
    else:
        n_barras = st.number_input("Numero total de barras (par)", value=8, min_value=4, step=2)
        R_efet = (D / 2) - dLinha
        dtheta = (2 * np.pi) / n_barras
        theta = np.pi / 2 + np.arange(n_barras) * dtheta
        ds = R_efet * np.sin(theta)
        ds = np.unique(np.round(ds, 6))
        n = np.ones_like(ds)
        if len(n) > 2:
            n[1:-1] = 2
        nLinha = len(n)

    modo_arm = st.radio("Definicao da bitola", ["Definir bitola manualmente",
                                                 "Dimensionar automaticamente (calcular As)"],
                         help="O dimensionamento automatico reproduz a formulacao do notebook "
                              "original, que pode nao convergir para todas as combinacoes de "
                              "N e M. Se isso ocorrer, a area minima sera usada e um aviso sera "
                              "exibido; recomenda-se conferir com a bitola manual.")
    bitola_manual = None
    if modo_arm == "Definir bitola manualmente":
        bitola_manual = st.selectbox("Bitola (mm)", BITOLAS_COMERCIAIS[:, 0].tolist(), index=2)

    st.header("5. Materiais")
    fck = st.number_input("fck (kN/m^2)", value=25000.0, step=1000.0)
    fyk = st.number_input("fyk (kN/m^2)", value=500000.0, step=10000.0)
    Es = st.number_input("Es - modulo de elasticidade do aco (kN/m^2)", value=200000000.0,
                          step=1000000.0, format="%.0f")
    usar_E_input = st.checkbox("Informar E do concreto manualmente (em vez de estimar por fck)",
                                value=False)
    E_concreto = 0.0
    if usar_E_input:
        E_concreto = st.number_input("E do concreto (kN/m^2)", value=24540000.0, step=100000.0)

    fcd = fck / 1.4
    fyd = fyk / 1.15
    eps0 = 0.002
    epsu = 0.0035
    k = 1 - (eps0 / epsu)

    st.header("6. Carregamentos (valores de calculo)")
    Nd = st.number_input("Nd - Forca normal de calculo (kN, compressao +)", value=1171.0, step=10.0)
    qd = st.number_input("qd - Carga distribuida de calculo (kN/m, + p/ direita)", value=0.0, step=1.0)

    if engastado:
        QTd = st.number_input("QTd - Forca transversal de calculo no topo (kN)", value=0.0, step=1.0)
        MTd = st.number_input("MTd - Momento de calculo no topo (kN.m)", value=28.69, step=1.0)
        MBd = 0.0
    else:
        MBd = st.number_input("MBd - Momento de calculo na base (kN.m)", value=0.0, step=1.0)
        MTd = st.number_input("MTd - Momento de calculo no topo (kN.m)", value=28.69, step=1.0)
        QTd = 0.0

    st.header("7. Impacto de veiculo (FEE)")
    categoria_impacto = st.selectbox(
        "Categoria do Veículo (Tabela 1 - NBR 6120)", 
        ["I", "II", "III", "IV", "V"]
    )
    direcao_impacto = st.radio("Direção da força", ["Frontal (Fx)", "Lateral (Fy)"], horizontal=True)
    
    dict_categorias = {
        "I": {"Fx": 100.0, "Fy": 50.0, "H": 0.5},
        "II": {"Fx": 180.0, "Fy": 90.0, "H": 0.5},
        "III": {"Fx": 240.0, "Fy": 120.0, "H": 1.0},
        "IV": {"Fx": 320.0, "Fy": 160.0, "H": 1.0},
        "V": {"Fx": 320.0, "Fy": 160.0, "H": 1.0},
    }
    
    valores_cat = dict_categorias[categoria_impacto]
    Fveic = valores_cat["Fx"] if "Frontal" in direcao_impacto else valores_cat["Fy"]
    Pos_Fveic = valores_cat["H"]
    
    st.info(f"**Força de impacto aplicada (Fveic):** {Fveic} kN\n\n**Altura de aplicação (Pos_Fveic):** {Pos_Fveic} m")

    # Parâmetros de 2a ordem / iteração fixados no código (removidos da interface gráfica)
    gamma_f3 = 1.1
    tol_iter = 1e-4
    max_iter = 100
    incremento_chi = 1e-5

    calcular = st.button("Calcular", type="primary", use_container_width=True)


# ============================================================
# 4. EXECUCAO DO PIPELINE DE CALCULO
# ============================================================

def run_pipeline():
    nNos = int(round(L / DeltaZ)) + 1

    # --- Rigidez linear constante ---
    if tipo_secao == "Retangular":
        r_param = 0
        b_param = b
        D_param = 0.0
    else:
        r_param = D / 2
        b_param = 0
        D_param = D

    coef_lin = 1.0
    EIef_scalar = rigidez_flexao_constante(b_param, h, r_param, fck,
                                            E_concreto if usar_E_input else 0.0, coef_lin)

    tamanho_vetor = nNos + 1 if engastado else nNos + 2
    EIef0 = EIef_scalar * np.ones(tamanho_vetor)

    # --- Analise Linear (P0) e Nao-linear geometrica (NLG) ---
    if engastado:
        W_P0 = deslocamento_lateral_engaste(nNos, DeltaZ, 0, qd, QTd, MTd, EIef0, Pos_Fveic, Fveic)
        M_P0 = momento_fletor_engaste(nNos, DeltaZ, EIef0, MTd, W_P0)
        W_NLG = deslocamento_lateral_engaste(nNos, DeltaZ, Nd, qd, QTd, MTd, EIef0, Pos_Fveic, Fveic)
        M_NLG = momento_fletor_engaste(nNos, DeltaZ, EIef0, MTd, W_NLG)
    else:
        W_P0 = deslocamento_lateral_biapoiados(nNos, DeltaZ, 0, qd, MBd, MTd, EIef0, Pos_Fveic, Fveic)
        M_P0 = momento_fletor_biapoiado(nNos, DeltaZ, EIef0, MBd, MTd, W_P0)
        W_NLG = deslocamento_lateral_biapoiados(nNos, DeltaZ, Nd, qd, MBd, MTd, EIef0, Pos_Fveic, Fveic)
        M_NLG = momento_fletor_biapoiado(nNos, DeltaZ, EIef0, MBd, MTd, W_NLG)

    # --- Dimensionamento da armadura longitudinal ---
    Md_dimensionamento = max(np.absolute(M_P0)) if not engastado else max(np.absolute(M_NLG))

    resultado_dim = {}
    if tipo_secao == "Retangular":
        x_raiz = fsolve(lambda x: equilibrio_forcas_e_momentos(
            x, b, h, ds, n, fcd, fyd, Es, eps0, epsu, k, Nd, Md_dimensionamento), h / 2)

        if modo_arm.startswith("Dimensionar"):
            Ast_Cal = area_aco_total(x_raiz, b, h, ds, n, fcd, fyd, Es, eps0, epsu, k, Nd,
                                      Md_dimensionamento)
            Ast_min = b * h * 0.004
            Ast_max = b * h * 0.04
            if (not np.isfinite(Ast_Cal)) or (Ast_Cal < 0):
                st.warning("O dimensionamento automatico nao convergiu para uma area de aco "
                           "valida nesta combinacao de N e M (limitacao da formulacao original "
                           "do notebook). Foi adotada a area de aco minima; recomenda-se conferir "
                           "com a opcao de bitola manual.")
                Ast_Cal = Ast_min
            elif Ast_Cal < Ast_min:
                Ast_Cal = Ast_min
            As_barra_Cal = Ast_Cal / np.sum(n)
            As_barra_Efet, BitolaFinal = bitola_comercial(As_barra_Cal)
            As = n * As_barra_Efet
            excedeu_max = np.sum(As) > Ast_max
        else:
            areas = BITOLAS_COMERCIAIS[:, 1]
            bitolas = BITOLAS_COMERCIAIS[:, 0]
            As_barra_Efet = areas[bitolas == bitola_manual][0]
            BitolaFinal = bitola_manual
            As = n * As_barra_Efet
            Ast_max = b * h * 0.04
            excedeu_max = np.sum(As) > Ast_max

        resultado_dim = dict(x_raiz=x_raiz, BitolaFinal=BitolaFinal, As=As,
                              As_total=np.sum(As), excedeu_max=excedeu_max)
    else:
        # Secao circular: segue o notebook original, que usa bitola pre-definida
        x_raiz = fsolve(lambda x: equilibrio_forcas_e_momentos(
            x, 0.0, h, ds, n, fcd, fyd, Es, eps0, epsu, k, Nd, Md_dimensionamento), h / 2)

        areas = BITOLAS_COMERCIAIS[:, 1]
        bitolas = BITOLAS_COMERCIAIS[:, 0]
        if modo_arm.startswith("Dimensionar"):
            st.warning("Para secao circular, o dimensionamento automatico nao e suportado de forma "
                       "totalmente rigorosa (o notebook original usa bitola pre-definida para "
                       "pilares circulares). Selecione uma bitola manualmente para maior precisao.")
            # aproxima usando area equivalente de secao retangular como fallback grosseiro
            b_eq = (np.pi * (D / 2) ** 2) / h
            Ast_Cal = area_aco_total(x_raiz, b_eq, h, ds, n, fcd, fyd, Es, eps0, epsu, k, Nd,
                                      Md_dimensionamento)
            Ast_min = ((np.pi * D ** 2) / 4) * 0.004
            if (not np.isfinite(Ast_Cal)) or (Ast_Cal < 0):
                st.warning("O dimensionamento automatico nao convergiu para uma area de aco "
                           "valida. Foi adotada a area de aco minima; recomenda-se usar a bitola "
                           "manual para secoes circulares.")
                Ast_Cal = Ast_min
            elif Ast_Cal < Ast_min:
                Ast_Cal = Ast_min
            As_barra_Cal = Ast_Cal / np.sum(n)
            As_barra_Efet, BitolaFinal = bitola_comercial(As_barra_Cal)
        else:
            As_barra_Efet = areas[bitolas == bitola_manual][0]
            BitolaFinal = bitola_manual

        As = n * As_barra_Efet
        Ast_max = ((np.pi * D ** 2) / 4) * 0.04
        excedeu_max = np.sum(As) > Ast_max
        resultado_dim = dict(x_raiz=x_raiz, BitolaFinal=BitolaFinal, As=As,
                              As_total=np.sum(As), excedeu_max=excedeu_max)

    As = resultado_dim["As"]

    # --- Diagrama N, M, 1/r ---
    dc = np.arange(-h / 2, h / 2, 0.01)
    dc = np.append(dc, h / 2)

    if tipo_secao == "Retangular":
        b_curva = b
    else:
        discre = dc
        b_curva = 2 * np.sqrt(np.maximum((D / 2) ** 2 - discre ** 2, 0.0))

    progress = st.progress(0.0, text="Gerando diagrama Momento-Curvatura (0.85 fcd)...")
    chi85, M85 = gerar_diagrama_mchi(b_curva, dc, ds, As, fcd, fyd, Es, eps0, epsu, Nd, 0.85,
                                      incremento_chi=incremento_chi,
                                      progress_cb=lambda p: progress.progress(p * 0.5))
    progress.progress(0.5, text="Gerando diagrama Momento-Curvatura (1.1 fcd)...")
    Mdbreak = M85[-1] / 1.1 if len(M85) else 0.0
    chi11, M11 = gerar_diagrama_mchi(b_curva, dc, ds, As, fcd, fyd, Es, eps0, epsu, Nd / 1.1, 1.1,
                                      incremento_chi=incremento_chi, Mdbreak=Mdbreak,
                                      progress_cb=lambda p: progress.progress(0.5 + p * 0.45))
    progress.progress(1.0, text="Diagrama concluido.")
    progress.empty()

    # --- Momento de fissuracao ---
    Mr = momento_fissuracao(fck, b if tipo_secao == "Retangular" else 0.0, h, D_param, x_raiz)

    # --- Iteracao NLG + NLF ---
    with st.spinner("Executando iteracao nao linear (NLG + NLF)..."):
        W_final, M_final, EIef_NL_final, logs, convergiu = iterar_nao_linear(
            nNos, DeltaZ, Nd, qd, QTd if engastado else MBd, MTd, EIef0, Pos_Fveic, Fveic,
            chi11, M11, Mr, engastado, gamma_f3=gamma_f3, tol=tol_iter, max_iter=int(max_iter))

    M_final_majorado = M_final * gamma_f3
    Mr_resistente = M85[-1] if len(M85) else np.nan

    return dict(
        nNos=nNos, EIef_scalar=EIef_scalar, W_P0=W_P0, M_P0=M_P0, W_NLG=W_NLG, M_NLG=M_NLG,
        chi85=chi85, M85=M85, chi11=chi11, M11=M11, Mr=Mr, x_raiz=x_raiz,
        resultado_dim=resultado_dim, W_final=W_final, M_final=M_final,
        M_final_majorado=M_final_majorado, EIef_NL_final=EIef_NL_final, logs=logs,
        convergiu=convergiu, Mr_resistente=Mr_resistente, ds=ds, n=n, As=As,
    )


if calcular:
    try:
        st.session_state["resultados"] = run_pipeline()
        st.session_state["params"] = dict(L=L, tipo_secao=tipo_secao, engastado=engastado)
    except Exception as e:
        st.error(f"Ocorreu um erro durante o calculo: {e}")
        st.exception(e)

if "resultados" in st.session_state:
    res = st.session_state["resultados"]
    p = st.session_state["params"]
    L_ = p["L"]

    tabs = st.tabs(["Resumo", "Diagrama M x chi", "Deslocamentos", "Momentos Fletores",
                    "Dimensionamento", "Fator de Seguranca de Impacto"])

    with tabs[0]:
        st.subheader("Resumo da entrada e da malha")
        c1, c2, c3 = st.columns(3)
        c1.metric("No de nos", res["nNos"])
        c2.metric("EI linear (kN.m2)", f"{res['EIef_scalar']:.2f}")
        c3.metric("Convergiu (NLG+NLF)?", "Sim" if res["convergiu"] else "Nao")
        st.write(f"Momento de fissuracao Mr = **{float(np.atleast_1d(res['Mr'])[0]):.3f} kN.m**")
        st.write(f"Momento resistente ultimo (0.85 fcd) = **{res['Mr_resistente']:.3f} kN.m**")

        df_logs = pd.DataFrame(res["logs"], columns=["Iteracao", "Residuo"])
        with st.expander("Historico de convergencia da iteracao nao linear"):
            st.dataframe(df_logs, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Diagrama Momento x Curvatura (N constante)")
        if len(res["chi85"]) == 0:
            st.warning("Nao foi possivel gerar a curva M-chi com os parametros informados.")
        else:
            fig = plot_momento_curvatura(res["chi85"], res["M85"], res["chi11"], res["M11"])
            st.pyplot(fig)

    with tabs[2]:
        st.subheader("Deslocamento lateral ao longo do pilar")
        series = [
            {"Z": res["W_P0"], "label": "Linear (LG e LF)"},
            {"Z": res["W_NLG"], "label": "Nao linear geometrica (NLG e LF)"},
            {"Z": res["W_final"], "label": "Nao linear (NLG e NLF)"},
        ]
        fig = plot_deslocamentos(series, L_)
        st.pyplot(fig)
        df_w = pd.DataFrame({
            "Posicao (m)": np.linspace(0, L_, len(res["W_final"])),
            "W linear (m)": res["W_P0"].flatten(),
            "W NLG (m)": res["W_NLG"].flatten(),
            "W NLG+NLF (m)": res["W_final"].flatten(),
        })
        st.dataframe(df_w, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Momento fletor ao longo do pilar")
        series = [
            {"M": res["M_P0"], "label": "Linear (LG e LF)"},
            {"M": res["M_NLG"], "label": "Nao linear geometrica (NLG e LF)"},
            {"M": res["M_final_majorado"], "label": "Nao linear (NLG e NLF) x gamma_f3"},
        ]
        fig = plot_momentos(series, L_)
        st.pyplot(fig)
        df_m = pd.DataFrame({
            "Posicao (m)": np.linspace(0, L_, len(res["M_final"])),
            "M linear (kN.m)": res["M_P0"],
            "M NLG (kN.m)": res["M_NLG"],
            "M NLG+NLF x gamma_f3 (kN.m)": res["M_final_majorado"],
        })
        st.dataframe(df_m, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Dimensionamento da armadura longitudinal")
        dim = res["resultado_dim"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Bitola adotada (mm)", f"{dim['BitolaFinal']}")
        c2.metric("As total (cm2)", f"{dim['As_total']*1e4:.2f}")
        c3.metric("Linha neutra x (m)", f"{float(np.atleast_1d(dim['x_raiz'])[0]):.4f}")
        if dim["excedeu_max"]:
            st.error("A area de aco excede a taxa maxima permitida (4%). Reavalie a secao.")
        df_as = pd.DataFrame({
            "Camada": np.arange(1, len(res["n"]) + 1),
            "Posicao ds (m)": res["ds"],
            "N. de barras": res["n"],
            "As da camada (cm2)": dim["As"] * 1e4,
        })
        st.dataframe(df_as, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Fator de Seguranca de Impacto (FSI)")
        df_fs, min_fs = tabela_fs(res["M_final_majorado"], res["Mr_resistente"])
        st.write(f"Momento fletor resistente: **{res['Mr_resistente']:.2f} kN.m**")
        st.write(f"Fator de seguranca minimo encontrado: **{min_fs:.2f}**")
        if min_fs < 1.0:
            st.error("FSI < 1: a secao nao resiste ao momento solicitante em pelo menos um ponto!")
        else:
            st.success("FSI >= 1 em toda a extensao do pilar.")
        st.dataframe(df_fs, use_container_width=True, hide_index=True)
        fig, _ = plot_fs(res["M_final_majorado"], res["Mr_resistente"], L_)
        st.pyplot(fig)
else:
    st.info("Preencha os parametros na barra lateral e clique em **Calcular** para executar a "
            "analise.")
