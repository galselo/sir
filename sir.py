# https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
#https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Total population, N.
N = int(6e7)

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# parameters
rnot = 3.7  # Rnot
gamma = 1. / 14. # recovery, 1/days
beta = rnot * gamma  # beta factor

# max days
xmax = 125.

# A grid of time points (in days)
t = np.linspace(0, xmax, 1000)

# load data
data = np.loadtxt("dpc-covid19-ita-andamento-nazionale.csv", delimiter=",").T

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
plot = ax.semilogy
plot(t, S, lw=2, label='contagiabili')
plot(t, I, lw=2, label='infetti')
plot(t, R, lw=2, label='immuni')

plot(t, I / 10., lw=2, label='TI (10% infetti)')
plt.hlines(5e3, 0, max(t))

# plot data
xdata = np.arange(len(data[5])) + 32
ydata = data[5]
plot(xdata, ydata, lw=0, marker="o", label="infetti (data)")
plot(xdata, data[2], lw=0, marker="x", label="TI (data)")

ax.set_xlabel('giorni')
ax.set_ylabel('individui')

# range
ax.set_xlim(xmin=20)
ax.set_ylim(ymin=200)

ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_locator(MultipleLocator(10))

legend = ax.legend(loc="best", ncol=2)

#plt.show()
plt.savefig("sir.png")


