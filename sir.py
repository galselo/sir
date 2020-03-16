# https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
# https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv
# https://github.com/pcm-dpc/COVID-19
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib


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
xmax = 110.

# time offset
toff = 44

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
#plot(t, S, lw=2, label='contagiabili')
#plot(t, I, lw=2, label='infetti*')
#plot(t, R, lw=2, label='immuni')

#plot(t, I * 0.008, lw=2, label='TI (0.8% infetti)')


# plot data
xdata = np.arange(len(data[9])) + toff
ydata = data[9]
plot(xdata, ydata, lw=0, marker="s", label="infetti* (data)", mfc='none')
plot(xdata, data[2], lw=0, marker="^", label="TI (data)", mfc='none')
plot(xdata, data[8], lw=0, marker="o", label="morti (data)", mfc='none')
#plot(xdata, data[8] + data[7], lw=0, marker="^", label="morti + guariti (data)", mfc='none')


cmap = matplotlib.cm.get_cmap('coolwarm')


nfit = 10

plist1 = []
for i in range(nfit):
    p = np.poly1d(np.polyfit(xdata[-5-i:-1-i], np.log10(data[2][-5-i:-1-i]), 1))
    plist1.append(p[1])
    plot(t, 1e1**p(t), lw=1, ls="--", c=cmap(i/(nfit - 1.)))

plist2 = []
for i in range(nfit):
    p = np.poly1d(np.polyfit(xdata[-5-i:-1-i], np.log10(ydata[-5-i:-1-i]), 1))
    plist2.append(p[1])
    plot(t, 1e1**p(t), lw=1, ls="--", c=cmap(i/(nfit - 1.)))

plist3 = []
for i in range(nfit):
    p = np.poly1d(np.polyfit(xdata[-5-i:-1-i], np.log10(data[8][-5-i:-1-i])-1, 1))
    plist3.append(p[1])
    plot(t, 1e1**p(t), lw=1, ls=":", c=cmap(i/(nfit - 1.)))



plt.hlines(5e3, 0, max(t), linestyle="-", color="#888a85", label="5k")
plt.hlines(7e3, 0, max(t), linestyle="--", color="#888a85", label="7k")
plt.hlines(2.8e4, 0, max(t), linestyle=":", color="#888a85", label="28k")

ymax = 1e9
plt.vlines(toff+5, 0, ymax, linestyle=":", color="#8ae234")
plt.vlines(toff+5+9, 0, ymax, linestyle="--", color="#8ae234")
plt.vlines(toff+5+31, 0, ymax, linestyle=":", color="#8ae234")
plt.vlines(toff+5+61, 0, ymax, linestyle=":", color="#8ae234")

ax.set_xlabel('giorni')
ax.set_ylabel('individui')

# range
ax.set_xlim(xmin=40, xmax=75)
ax.set_ylim(ymin=5, ymax=5e4)

ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_locator(MultipleLocator(10))

legend = ax.legend(loc="best", ncol=2, fontsize=6)

#plt.show()
plt.savefig("sir.png")

plt.clf()
plt.plot(range(len(plist1)), plist1, marker="o", label="TI")
plt.plot(range(len(plist2)), plist2, marker="o", label="infetti")
plt.plot(range(len(plist3)), plist3, marker="o", label="morti")
plt.gca().invert_xaxis()
plt.xlabel("- giorni")
plt.ylabel("pendenza")
plt.legend(loc="best")
plt.savefig("coef.png")


