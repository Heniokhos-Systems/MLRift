import numpy as np
DT,TAU_M,V_REST,V_THRESH,V_RESET,REFRAC=0.1,10.0,0.0,1.0,0.0,20
N,STEPS=5,300
theta=np.array([-0.03,-0.01,0.0,0.02,0.04]); drive=np.array([1.4,1.6,1.55,1.5,1.45])
V=np.full(N,V_REST); refr=np.zeros(N,int); cnt=np.zeros(N,int); vth=V_THRESH+theta
for _ in range(STEPS):
    active=refr<=0
    a=V_REST-V[active]; a=a+drive[active]; a=DT*a; a=a/TAU_M; V[active]=V[active]+a
    sp=active&(V>=vth); cnt[sp]+=1; V[sp]=V_RESET; refr[sp]=REFRAC; refr[~active]-=1
for i in range(N): print(f"{i} {cnt[i]}")   # integer spike counts — byte-diffable
