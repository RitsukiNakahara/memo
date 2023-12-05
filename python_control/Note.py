#%%
from control import tf,ss,series,feedback

s1=tf([0,1],[1,1])
s2=tf([0,1],[1,2])
s3=tf([3,1],[1,0])
s4=tf([2,0],[0,1])

s12=feedback(s1,s2)
s123=series(s12,s3)
s=feedback(s123,s4)
print("S=",s)

#%%
def plot_set(fig_ax,*args):
    fig_ax.set_xlabel(args[0])
    fig_ax.set_ylabel(args[1])
    fig_ax.grid(ls=":")
    if len(args)==3:
        fig_ax.legend(loc=args[2])

# %%
from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np

T,K=0.5,1
P=tf([0,K],[T,1])
y,t=step(P,np.arange(0,5,0.01))

fig,ax=plt.subplots()
ax.plot(t,y)
plot_set(ax,"t","y")

# %%
def linestyle_generator():
    linestyle=["-","--","-.",":"]
    lineID=0
    while True:
        yield linestyle[lineID]
        lineID=(lineID+1)%len(linestyle)

# %%
fig,ax=plt.subplots()
LS=linestyle_generator()
K=1
T=[1,0.5,0.1]
for i in range(len(T)):
    y,t=step(tf([0,K],[T[i],1]),np.arange(0,5,0.01))
    ax.plot(t,y,ls=next(LS),label="T="+str(T[i]))

plot_set(ax,"t","y","best")

# %%
fig,ax=plt.subplots()
LS=linestyle_generator()
T=0.5
K=[1,2,3]
for i in range(len(K)):
    y,t=step(tf([0,K[i]],[T,1]),np.arange(0,5,0.01))
    ax.plot(t,y,ls=next(LS),label="K="+str(K[i]))

plot_set(ax,"t","y","upper left")

# %%
import sympy as sp
sp.init_printing()
s=sp.Symbol("s")
T=sp.Symbol("T",real=True)
P=1/((1+T*s)*s)
Q=sp.apart(P,s)
print(Q)

t=sp.Symbol("t",positive=True)
q=sp.inverse_laplace_transform(Q,s,t)
print(q)

# %%
zeta,omega_n=0.4,5

P=tf([0,omega_n**2],[1,2*zeta*omega_n,omega_n**2])
y,t=step(P,np.arange(0,5,0.01))

fig,ax=plt.subplots()
ax.plot(t,y)
plot_set(ax,"t","y")

# %%
LS=linestyle_generator()
fig,ax=plt.subplots()

zeta=[1,0.7,0.4]
omega_n=5

for i in range(len(zeta)):
    P=tf([0,omega_n**2],[1,2*zeta[i]*omega_n,omega_n**2])
    y,t=step(P,np.arange(0,5,0.01))

    pltargs={"ls":next(LS)}
    pltargs["label"]="$\zeta$="+str(zeta[i])
    ax.plot(t,y,**pltargs)

plot_set(ax,"t","y","best")

# %%
LS=linestyle_generator()
fig,ax=plt.subplots()

zeta=0.4
omega_n=[1,5,10]

for i in range(len(omega_n)):
    P=tf([0,omega_n[i]**2],[1,2*zeta*omega_n[i],omega_n[i]**2])
    y,t=step(P,np.arange(0,5,0.01))

    pltargs={"ls":next(LS)}
    pltargs["label"]="$\omega_n$="+str(omega_n[i])
    ax.plot(t,y,**pltargs)

plot_set(ax,"t","y","best")

# %%
