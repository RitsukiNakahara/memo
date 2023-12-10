#%%　ブロック線図
from control import tf,ss,series,feedback

s1=tf([0,1],[1,1])
s2=tf([0,1],[1,2])
s3=tf([3,1],[1,0])
s4=tf([2,0],[0,1])

s12=feedback(s1,s2)
s123=series(s12,s3)
s=feedback(s123,s4)
print("S=",s)

#%%　グラフ整形
def plot_set(fig_ax,*args):
    fig_ax.set_xlabel(args[0])
    fig_ax.set_ylabel(args[1])
    fig_ax.grid(ls=":")
    if len(args)==3:
        fig_ax.legend(loc=args[2])

# %%　1次遅れ系
from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np

T,K=0.5,1
P=tf([0,K],[T,1])
y,t=step(P,np.arange(0,5,0.01))

fig,ax=plt.subplots()
ax.plot(t,y)
plot_set(ax,"t","y")

# %%　グラフ整形
def linestyle_generator():
    linestyle=["-","--","-.",":"]
    lineID=0
    while True:
        yield linestyle[lineID]
        lineID=(lineID+1)%len(linestyle)

# %%　1次遅れ系(T)
fig,ax=plt.subplots()
LS=linestyle_generator()
K=1
T=[1,0.5,0.1]
for i in range(len(T)):
    y,t=step(tf([0,K],[T[i],1]),np.arange(0,5,0.01))
    ax.plot(t,y,ls=next(LS),label="T="+str(T[i]))

plot_set(ax,"t","y","best")

# %%　1次遅れ系(K)
fig,ax=plt.subplots()
LS=linestyle_generator()
T=0.5
K=[1,2,3]
for i in range(len(K)):
    y,t=step(tf([0,K[i]],[T,1]),np.arange(0,5,0.01))
    ax.plot(t,y,ls=next(LS),label="K="+str(K[i]))

plot_set(ax,"t","y","upper left")

# %%  逆ラプラス変換
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

# %%　2次遅れ系
zeta,omega_n=0.4,5

P=tf([0,omega_n**2],[1,2*zeta*omega_n,omega_n**2])
y,t=step(P,np.arange(0,5,0.01))

fig,ax=plt.subplots()
ax.plot(t,y)
plot_set(ax,"t","y")

# %%　2次遅れ系
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

# %%　2次遅れ系
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

# %%  ステップ応答練習問題
P=tf([1,3],[1,3,2])
y,t=step(P,np.arange(0,5,0.01))

fig,ax=plt.subplots()
ax.plot(t,y)
plot_set(ax,"t","y")

# %%　ステップ応答練習問題
P=tf([0,1],[1,2,2,1])
y,t=step(P,np.arange(0,5,0.01))

fig,ax=plt.subplots()
ax.plot(t,y)
plot_set(ax,"t","y")

# %% 状態空間モデルの時間応答
A=[[0,1],[-4,-5]]
B=[[0],[1]]
C=np.eye(2)
D=np.zeros([2,1])
P=ss(A,B,C,D)
Td=np.arange(0,5,0.01)
Ud=1*(Td>0)

xst,t=step(P,Td)
xin,t=initial(P,Td,-0.3)
x,_,_=lsim(P,Ud,Td,-0.3)

fig,ax=plt.subplots()
ax.plot(t,x[:,0],label="response")
ax.plot(t,xst[:,0],ls="--",label="zero state")
ax.plot(t,xin[:,0],ls="-.",label="zero input")

plot_set(ax,"t","x","best")

# %%　状態空間モデル時間応答問題
A=[[0,1],[-4,-5]]
B=[[0],[1]]
C=np.eye(2)
D=np.zeros([2,1])
P=ss(A,B,C,D)
Td=np.arange(0,5,0.01)
Ud=3*np.sin(5*Td)

xst,t=step(P,Td)
xin,t=initial(P,Td,0.5)
x,_,_=lsim(P,Ud,Td,0.5)

fig,ax=plt.subplots()
ax.plot(t,x[:,0],label="response")
ax.plot(t,xst[:,0],ls="--",label="zero state")
ax.plot(t,xin[:,0],ls="-.",label="zero input")

plot_set(ax,"t","x","best")

# %%  漸近安定性の判定
w=1.5
Y,X=np.mgrid[-w:w:100j,-w:w:100j]

A=np.array([[0,1],[-4,5]])
s,v=np.linalg.eig(A)

U=A[0,0]*X+A[0,1]*Y
V=A[1,0]*X+A[1,1]*Y

t=np.arange(-1.5,1.5,0.01)

fig,ax=plt.subplots()

if s.imag[0]==0 and s.imag[1]==0:
    ax.plot(t,v[1,0]/v[0,0]*t,ls="-")
    ax.plot(t,v[1,1]/v[1,1]*t,ls="-")

ax.streamplot(X,Y,U,V,density=0.7,color="k")
plot_set(ax,"$x_1$","x_2")

print(s)

#%%  ボード線図
def bodeplot_set(fig_ax,*args):
    fig_ax[0].grid(which="both",ls=":")
    fig_ax[0].set_ylabel("Gain [dB]")
    fig_ax[1].grid(which="both",ls=":")
    fig_ax[1].set_xlabel("$\omega$ [rad/s]")
    fig_ax[1].set_ylabel("Phase [deg]")

    if len(args)>0:
        fig_ax[1].legend(loc=args[0])
    if len(args)>1:
        fig_ax[0].legend(loc=args[1])

# %%　ボード線図
K=1
T=[1,0.5,0.1]

LS=linestyle_generator()
fig,ax=plt.subplots(2,1)
for i in range(len(T)):
    P=tf([0,K],[T[i],1])
    gain,phase,w=bode(P,logspace(-2,2),plot=False)
    pltargs={"ls":next(LS),"label":"T="+str(T[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,3,3)

# %%　ボード線図(2次遅れ系)
zeta=[1,0.7,0.4]
omega_n=1

LS=linestyle_generator()
fig,ax=plt.subplots(2,1)
for i in range(len(zeta)):
    P=tf([0,omega_n**2],[1,2*zeta[i]*omega_n,omega_n**2])
    gain,phase,w=bode(P,logspace(-2,2),plot=False)
    pltargs={"ls":next(LS),"label":"$\zeta$="+str(zeta[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,3,3)

# %% ボード線図練習問題
LS=linestyle_generator()
fig,ax=plt.subplots(2,1)
P=tf([1,3],[1,3,2])
gain,phase,w=bode(P,logspace(-2,2),plot=False)
pltargs={"ls":next(LS)}
ax[0].semilogx(w,20*np.log10(gain),**pltargs)
ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,1,1)

# %%　ボード線図練習問題
LS=linestyle_generator()
fig,ax=plt.subplots(2,1)
P=tf([0,1],[1,2,2,1])
gain,phase,w=bode(P,logspace(-2,2),plot=False)
pltargs={"ls":next(LS)}
ax[0].semilogx(w,20*np.log10(gain),**pltargs)
ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,1,1)

# %%　P制御
g=9.81
l=0.2
M=0.5
mu=1.5e-2
J=1.0e-2

P=tf([0,1],[J,mu,M*g*l])

ref=30

kp=(0.5,1,2)

LS=linestyle_generator()
fig,ax=plt.subplots()
for i in range(3):
    K=tf([0,kp[i]],[0,1])
    Gyr=feedback(P*K,1)
    y,t=step(Gyr,np.arange(0,2,0.01))

    pltargs={"ls":next(LS),"label":"$k_p=$"+str(kp[i])}
    ax.plot(t,y*ref,**pltargs)

ax.axhline(ref,color="k",linewidth=0.5)
plot_set(ax,"t","y","best")

# %% P制御ボード線図
LS=linestyle_generator()
fig,ax=plt.subplots(2,1)

for i in range(3):
    K=tf([0,kp[i]],[0,1])
    Gyr=feedback(P*K,1)

    gain,phase,w=bode(Gyr,logspace(-1,2),plot=False)
    pltargs={"ls":next(LS),"label":"$k_p=$"+str(kp[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,"lower left")

# %%　PD制御
kp=2
kd=(0,0.1,0.2)

LS=linestyle_generator()
fig,ax=plt.subplots()
for i in range(3):
    K=tf([kd[i],kp],[0,1])
    Gyr=feedback(P*K,1)
    y,t=step(Gyr,np.arange(0,2,0.01))

    pltargs={"ls":next(LS),"label":"$k_d=$"+str(kd[i])}
    ax.plot(t,y*ref,**pltargs)

ax.axhline(ref,color="k",linewidth=0.5)
plot_set(ax,"t","y","best")

# %%　PD制御ボード線図
LS=linestyle_generator()
fig,ax=plt.subplots(2,1)

for i in range(3):
    K=tf([kd[i],kp],[0,1])
    Gyr=feedback(P*K,1)

    gain,phase,w=bode(Gyr,logspace(-1,2),plot=False)
    pltargs={"ls":next(LS),"label":"$k_d=$"+str(kd[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,"lower left")

# %%　PID制御
kp=2
kd=0.1
ki=(0,5,10)

LS=linestyle_generator()
fig,ax=plt.subplots()
for i in range(3):
    K=tf([kd,kp,ki[i]],[1,0])
    Gyr=feedback(P*K,1)
    y,t=step(Gyr,np.arange(0,2,0.01))

    pltargs={"ls":next(LS),"label":"$k_i=$"+str(ki[i])}
    ax.plot(t,y*ref,**pltargs)

ax.axhline(ref,color="k",linewidth=0.5)
plot_set(ax,"t","y","upper left")

# %%　PID制御ボード線図
LS=linestyle_generator()
fig,ax=plt.subplots(2,1)

for i in range(3):
    K=tf([kd,kp,ki[i]],[1,0])
    Gyr=feedback(P*K,1)

    gain,phase,w=bode(Gyr,logspace(-1,2),plot=False)
    pltargs={"ls":next(LS),"label":"$k_i=$"+str(ki[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,"best")

# %%　PID制御練習問題
kp=2
kd=0.1
ki=(0,5,10)

LS=linestyle_generator()
fig,ax=plt.subplots()
for i in range(3):
    K=tf([kd,kp,ki[i]],[1,0])
    Gyd=feedback(P,K)
    y,t=step(Gyd,np.arange(0,2,0.01))

    pltargs={"ls":next(LS),"label":"$k_i=$"+str(ki[i])}
    ax.plot(t,y,**pltargs)

plot_set(ax,"t","y","center right")

# %% PID制御練習問題
LS=linestyle_generator()
fig,ax=plt.subplots(2,1)

for i in range(3):
    K=tf([kd,kp,ki[i]],[1,0])
    Gyd=feedback(P,K)

    gain,phase,w=bode(Gyd,logspace(-1,2),plot=False)
    pltargs={"ls":next(LS),"label":"$k_i=$"+str(ki[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

bodeplot_set(ax,"best")

# %%　改良型PID制御
kp=2
kd=0.1
ki=10

K1=tf([kd,kp,ki],[1,0])
K2=tf([0,ki],[kd,kp,ki])

Gyz=feedback(P*K1,1)

Td=np.arange(0,2,0.01)
r=1*(Td>0)

z,t,_=lsim(K2,r,Td,0)
fig,ax=plt.subplots(1,2)

y,_,_=lsim(Gyz,r,Td,0)
ax[0].plot(t,r*ref,color="k")
ax[1].plot(t,y*ref,ls="--",label="PID",color="k")

y,_,_=lsim(Gyz,z,Td,0)
ax[0].plot(t,z*ref,color="k")
ax[1].plot(t,y*ref,label="PI-D",color="k")

ax[1].axhline(ref,color="k",linewidth=0.5)
plot_set(ax[0],"t","r")
plot_set(ax[1],"t","r","best")

# %%　限界感度法
num_delay,den_delay=pade(0.005,1)
Pdelay=P*tf(num_delay,den_delay)

fig,ax=plt.subplots()

kp0=2.9
K=tf([0,kp0],[0,1])
Gyr=feedback(Pdelay*K,1)
y,t=step(Gyr,np.arange(0,2,0.01))

ax.plot(t,y*ref,color="k")
ax.axhline(ref,color="k",linewidth=0.5)
plot_set(ax,"t","y")

# %%　ゲインチューニング
T0=0.3

Rule="No Overshoot"
kp=0.2*kp0
ki=kp/(0.5*T0)
kd=kp*(0.33*T0)

LS=linestyle_generator()
fig,ax=plt.subplots()

K=tf([kd,kp,ki],[1,0])
Gyr=feedback(Pdelay*K,1)
y,t=step(Gyr,np.arange(0,2,0.01))

ax.plot(t,y*ref,ls=next(LS),label=Rule)

ax.axhline(ref,color="k",linewidth=0.5)
plot_set(ax,"t","y","best")

# %%　モデルマッチング法
import sympy as sp
s=sp.Symbol("s")
kp,kd,ki=sp.symbols("k_p k_d k_i")
Mgl,mu,J=sp.symbols("Mgl,mu,J")
sp.init_printing()

G=(kp*s+ki)/(J*s**3+(mu+kd)*s**2+(Mgl+kp)*s+ki)
sp.series(1/G,s,0,3)

# %%
wn=sp.symbols("omega_n")
M=wn**2/(s**2+2*wn*s+wn**2)
sp.series(1/M,s,0,3)

# %%　
z,wn=sp.symbols("zeta omega_n")
kp,kd,ki=sp.symbols("k_p k_d k_i")
Mgl,mu,J=sp.symbols("Mgl,mu,J")
sp.init_printing()

f1=Mgl/ki-2*z/wn
f2=(mu+kd)/ki-Mgl*kp/(ki**2)-1/(wn**2)
f3=J/ki-kp*(mu+kd)/(ki**2)+Mgl*kp**2/(ki**3)
sp.solve([f1,f2,f3],[kp,kd,ki])

# %%　ゲインチューニング
#エラー

# %%　モデルマッチング法練習問題
import sympy as sp
s=sp.Symbol("s")
kp,kd,ki=sp.symbols("k_p k_d k_i")
Mgl,mu,J=sp.symbols("Mgl,mu,J")
sp.init_printing()

G=(ki)/(J*s**3+(mu+kd)*s**2+(Mgl+kp)*s+ki)
sp.series(1/G,s,0,4)

#%% 
wn=sp.symbols("omega_n")
M=wn**3/(s**3+3*wn*s**2+3*wn**2*s+wn**3)
sp.series(1/M,s,0,4)

#%%
z,wn=sp.symbols("zeta omega_n")
kp,kd,ki=sp.symbols("k_p k_d k_i")
Mgl,mu,J=sp.symbols("Mgl,mu,J")
sp.init_printing()

f1=(Mgl+kp)/ki-3/wn
f2=(kd+mu)/ki-3/(wn**2)
f3=J/ki-1/(wn**3)
sp.solve([f1,f2,f3],[kp,kd,ki])

# %%　極配置法
A="0 1;-4 5"
B="0;1"
C="1 0;0 1"
D="0;0"
P=ss(A,B,C,D)

Pole=[-1,-1]
F=-acker(P.A,P.B,Pole)

Acl = P.A + P.B*F
Pfb = ss(Acl, P.B, P.C, P.D)

Td = np.arange(0, 5, 0.01)
X0 = [-0.3, 0.4]
x, t = initial(Pfb, Td, X0)

fig, ax = plt.subplots(figsize=(3, 2.3))
ax.plot(t, x[:,0], label = '$x_1$')
ax.plot(t, x[:,1], ls = '-.', label = '$x_2$')

plot_set(ax, 't', 'x', 'best')

# %%　最適レギュレータ
Q=[[100,0],[0,1]]
R=1

F,X,E=lqr(P.A,P.B,Q,R)
F=-F

Acl = P.A + P.B*F
Pfb = ss(Acl, P.B, P.C, P.D)

Td = np.arange(0, 5, 0.01)
X0 = [-0.3, 0.4]
x, t = initial(Pfb, Td, X0)

fig, ax = plt.subplots(figsize=(3, 2.3))
ax.plot(t, x[:,0], label = '$x_1$')
ax.plot(t, x[:,1], ls = '-.', label = '$x_2$')

plot_set(ax, 't', 'x', 'best')

# %%　開ループ系の安定性　
P=tf([0,1],[1,1,1.5,1])
_,_,wpc,_=margin(P)

t=np.arange(0,30,0.1)
u=np.sin(wpc*t)
y=0*u

fig,ax=plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        u=np.sin(wpc*t)-y
        y,t,x0=lsim(P,u,t,0)
        
        ax[i,j].plot(t,u,ls="--",label="u")
        ax[i,j].plot(t,y,label="y")
        plot_set(ax[i,j],"t","u,y",)

# %%
P=tf([0,1],[1,2,2,1])
_,_,wpc,_=margin(P)

t=np.arange(0,30,0.1)
u=np.sin(wpc*t)
y=0*u

fig,ax=plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        u=np.sin(wpc*t)-y
        y,t,x0=lsim(P,u,t,0)
        
        ax[i,j].plot(t,u,ls="--",label="u")
        ax[i,j].plot(t,y,label="y")
        plot_set(ax[i,j],"t","u,y",)

# %%　ナイキスト軌跡
fig,ax=plt.subplots(1,2)

P=tf([0,1],[1,1,1.5,1])
x,y,_=nyquist(P,logspace(-3,5,1000),plot=False)
ax[0].plot(x,y,color="k")
ax[0].plot(x,-y,ls="--",color="k")
ax[0].scatter(-1,0,color="k")
plot_set(ax[0],"Re","Im")

P=tf([0,1],[1,2,2,1])
x,y,_=nyquist(P,logspace(-3,5,1000),plot=False)
ax[1].plot(x,y,color="k")
ax[1].plot(x,-y,ls="--",color="k")
ax[1].scatter(-1,0,color="k")
plot_set(ax[1],"Re","Im")

fig.tight_layout()

# %%　開ループ系P制御
g  = 9.81                
l  = 0.2                 
M  = 0.5                 
mu = 1.5e-2              
J  = 1.0e-2              

P = tf( [0,1], [J, mu, M*g*l] )

ref = 30

kp=(0.5,1,2)

LS=linestyle_generator()
fig,ax=plt.subplots(2,1)
for i in range(3):
    K=tf([0,kp[i]],[0,1])
    H=P*K
    gain,phase,w=bode(H,logspace(-1,2),plot=False)

    pltargs={"ls":next(LS),"label":"$k_p=$"+str(kp[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

    print('kP=', kp[i])
    print('(GM, PM, wpc, wgc)')
    print(margin(H))
    print('-----------------')
    
bodeplot_set(ax, 3)

# %%　開ループ系PI制御
kp=2
ki=(0,5,10)

LS=linestyle_generator()
fig,ax=plt.subplots(2,1)
for i in range(3):
    K=tf([kp,ki[i]],[1,0])
    H=P*K
    gain,phase,w=bode(H,logspace(-1,2),plot=False)

    pltargs={"ls":next(LS),"label":"$k_i$="+str(ki[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

    print('ki=', ki[i])
    print('(GM, PM, wpc, wgc)')
    print(margin(H))
    print('-----------------')
    
bodeplot_set(ax, 3)

# %%　開ループ系PID制御
kp=2
ki=5
kd=(0,0.1,0.2)

LS=linestyle_generator()
fig,ax=plt.subplots(2,1)
for i in range(3):
    K=tf([kd[i],kp,ki],[1,0])
    H=P*K
    gain,phase,w=bode(H,logspace(-1,2),plot=False)

    pltargs={"ls":next(LS),"label":"$k_d=$"+str(kd[i])}
    ax[0].semilogx(w,20*np.log10(gain),**pltargs)
    ax[1].semilogx(w,phase*180/np.pi,**pltargs)

    print('kd=', kd[i])
    print('(GM, PM, wpc, wgc)')
    print(margin(H))
    print('-----------------')
    
bodeplot_set(ax, 3)

# %%　閉ループ系ステップ応答
LS=linestyle_generator()
fig,ax=plt.subplots()
for i in range(3):
    K=tf([kd[i],kp,ki],[1,0])
    Gyd=feedback(P*K,1)
    y,t=step(Gyd,np.arange(0,2,0.01))

    pltargs={"ls":next(LS),"label":"$k_d=$"+str(kd[i])}
    ax.plot(t,y*ref,**pltargs)

ax.axhline(ref,color="k",linewidth=0.5)
plot_set(ax,"t","y",4)

# %%
