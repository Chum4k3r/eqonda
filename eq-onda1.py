#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:50:20 2020

@author: joaovitor
"""

import time
import multiprocessing as mp
import numpy as np
from scipy import signal as ss
from matplotlib import pyplot as plt
#import sounddevice as sd


# 1. Variáveis e álgebra
fs = 44100  # taxa de amostragem [Hz]
dt = 1/fs  # intervalo de tempo entre amostras [s]
nsamp = 2**9  # número total de amostras [-]
tlen = (nsamp - 1) / fs  # duração do sinal [s]

print (f"{fs=}    {dt=}\n{nsamp=}    {tlen=}")


b = 0  # Número de semitons  [-]
c = 343  # Velocidade de propagação do som no ar [m/s]
f = 440 * 2**(b/12)  # frequência [Hz]
T = 1/f  #  Período [s]
l = c / f  # comprimento de onda [m]

print(f"{c=}    {f=}\n{T=}    {l=}")



omega = 2 * np.pi * f  # frequencia angular [rad/s]
k = 2 * np.pi / l  # número de onda [rad/m]
kl = 1/l

print (f"{omega=}    {k=}    {kl=}")




nmark = 2 * nsamp  # número de marcações no espaço [-]
dx = 1 / nmark  # intervalo entre as marcações [m]



print (f"{nmark=}    {dx=}")





xm = np.linspace(0, 1 - dx, nmark)  # [m] marcações no espaço
ts = np.linspace(0, tlen, nsamp)  # [s] instantes de tempo

cosx = np.exp(k * xm * 1j)
cost = np.exp(-omega * ts * 1j)
"""
Explique o que está sendo calculado nas linhas 59 e 60:

    Escreva aqui

"""


print (f"{xm.shape=}    {cosx.shape=}")
print (f"{ts.shape=}    {cost.shape=}")


fig1, axs = plt.subplots(2, 1)

xlines = []
xlines.append(*axs[0].plot(xm, np.real(cosx)))
xlines.append(*axs[0].plot(xm, np.imag(cosx)))
xlines.append(axs[0].hlines(np.abs(cosx)/(2**0.5), xm[0], xm[-1]))

tlines = []
tlines.append(*axs[1].plot(ts, np.real(cost)))
tlines.append(*axs[1].plot(ts, np.imag(cost)))
tlines.append(axs[1].hlines(np.abs(cost)/(2**0.5), ts[0], ts[-1]))

axs[0].set_title("Gráfico de Re{exp(jkx)} variando no tempo")
axs[0].set_xlabel("Posição [m]")
axs[0].set_ylabel("Amplitude [-]")

axs[1].set_title("Gráfico de Re{exp(-jωt)} variando no espaço")
axs[1].set_xlabel("Tempo [s]")
axs[1].set_ylabel("Amplitude [-]")

fig1.tight_layout()
fig1.canvas.draw()
fig1.show()


def loop(fig1, axs, xlines, tlines, cosx, cost, nsamp):
    from time import time
    tstart = time()
    for n in range(nsamp):
        extn = cosx * cost[n]
        exnt = cosx[2*n] * cost

        xlines[0].set_ydata(np.real(extn))
        xlines[1].set_ydata(np.imag(extn))
        tlines[0].set_ydata(np.real(exnt))
        tlines[1].set_ydata(np.imag(exnt))

        axs[0].draw_artist(axs[0].patch)
        [axs[0].draw_artist(xline) for xline in xlines]

        axs[1].draw_artist(axs[1].patch)
        [axs[1].draw_artist(tline) for tline in tlines]

        fig1.canvas.update()
        fig1.canvas.flush_events()
    tend = time()
    return tend - tstart

tloop = loop(fig1, axs, xlines, tlines, cosx, cost, nsamp)
print(f"{tloop=:.6f}")



def rms(x: np.ndarray):
    return (x**2).mean(axis=0)**0.5


def dB(x: np.ndarray):
    return 20*np.log10(rms(x))


print(f"{rms(np.real(cosx))=:.3f}  {dB(np.real(cosx))=:.1f}")
print(f"{rms(np.real(cost))=:.3f}  {dB(np.real(cost))=:.1f}")



cos_xt = np.zeros((xm.size, ts.size), dtype='complex64')

print (f"{cos_xt.shape=}")


print(f"começo do cálculo de exp(jkx - jωt)")

tstart = time.time()

for n, x in enumerate(xm):

    cos_xt[n, :] = np.exp(1j*k*x - 1j*omega*ts)
    # print(f"{n=}  {x=:.2f}  {dB(np.real(cos_xt[n, :]))=:.1f}",
    #       end=('\t' if (n+1) % 3 else '\n'))  # operador ternário
    #            # expressão diz que:
    #            # Se a sobra inteira da divisão de n por três  for diferente de zero,
    #            # a mensagem deve terminar com uma tabulação.
    #            # Senão, termina com nova linha.

tend = time.time()

print(f"Complete in {tend - tstart:.6f} seconds.")



fig2, ax = plt.subplots(1, 1)
im = ax.pcolormesh(ts, xm, np.real(cos_xt), cmap='jet')

ax.set_title("Gráfico de cos(kx - ωt)")
ax.set_ylabel("Posição [m]")
ax.set_xlabel("Tempo [s]")
fig2.colorbar(im)

fig2.tight_layout()
fig2.show()



fig3, ax3 = plt.subplots(1, 1)
im3 = ax3.pcolormesh(ts, xm, np.imag(cos_xt), cmap='jet')

ax3.set_title("Gráfico de cos(kx - ωt)")
ax3.set_ylabel("Posição [m]")
ax3.set_xlabel("Tempo [s]")
fig3.colorbar(im3)

fig3.tight_layout()
fig3.show()

input("'Enter' para sair.")


