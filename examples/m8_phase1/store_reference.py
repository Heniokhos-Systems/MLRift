import numpy as np
A,N_SYM,K_SYM,Mc,K_CTX,L=3,6,2,8,3,5
sym_code=np.array([[0,1],[2,3],[4,5]]); seq=np.array([0,1,2,0,1])
cue_code=np.array([[0,1,2],[2,3,4],[4,5,6],[1,3,5],[0,2,4]])
W=np.zeros((N_SYM,Mc),int)
for t in range(L): W[np.ix_(sym_code[seq[t]],cue_code[t])]+=1
cue_idx=np.array([0,2,4]); raw=W[:,cue_idx].sum(1)
np.savetxt("store_seq.txt",seq,fmt="%d"); np.savetxt("store_symcode.txt",sym_code,fmt="%d")
np.savetxt("store_cuecode.txt",cue_code,fmt="%d"); np.savetxt("store_cueidx.txt",cue_idx,fmt="%d")
for i in range(N_SYM): print(f"{i} {raw[i]}")
