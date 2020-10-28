import numpy as np
from dsGameSolver.gameSolver import dsSolve
import scipy
from scipy.stats import uniform
import math
from scipy.stats import triang

import itertools



def Cross_product2(A,B):
    
    L=[]
    for i in range(len(A)):
        for j in range(len(B)):
            l=[A[i],B[j]]
            L.append(l)
    return(L)   

def Cross_product(A,B,C):
    
    L=[]
    for i in range(len(A)):
        for j in range(len(B)):
            for k in range(len(C)):
                l=[A[i],B[j],C[k]]
                L.append(l)
    return(L)    

b=3#suppport of consumer preferences
def market_share(l1,l2,adv_left):
    tau=1
     
    if l1<l2:
        x_hat=0.5*(l1+l2)+(adv_left/(tau*(l2-l1)))
        s=np.round(uniform.cdf(x_hat,0,b),3) #rounding upto 2
    else:
        if l1>l2:
            x_hat=0.5*(l2+l1)+((-1*adv_left)/(tau*(l1-l2)))
            s=1-np.round(uniform.cdf(x_hat,0,b),3)
        else:
            if adv_left>0:
                s=1
            elif adv_left==0:
                s=0.5
            else:
                s=0
    return(np.round(s,5))           


a_beg=0
a_end=b
num_actions = 3
A= np.linspace(a_beg, a_end,num_actions )

Positions=Cross_product2(A,A)



max_adv=0.1
eh=np.round(0.8*max_adv,3)
em=np.round(0.3*max_adv,3)
noise=np.round(0.2*max_adv,3)

E=[0,eh]
N=[-noise,noise]

Advantages=[]
for i in range(len(E)):
    for j in range(len(E)):
        for k in range(len(N)):
            a=E[i]-E[j]+N[k]
            Advantages.append(a)


States_E=Advantages
States_E=set(States_E)
print(States_E)
share=[]

ADV=list(States_E)
Pre_E=Cross_product2(Positions,ADV)

share_new=[]

for i in range(len(Pre_E)):
    triple=Pre_E[i]
    pos=triple[0]
    adv_new=triple[1]
    a1=pos[0]
    a2=pos[1]
    s=market_share(a1,a2,adv_new)
    share_new.append(s)


for i in range(len(A)):
    for j in range(len(A)):
        for adv in range(len(ADV)):
            s=market_share(A[i],A[j],ADV[adv])
            share.append(s)




Shares=set(share_new)
Shares=list(Shares)

States_G=Shares


def U1_E(s1,l1):
    p1=0.5
    U=s1-(p1-l1)**2
    return(np.round(U,4))
def U2_E(s2,l2):
    p2=2;
    U=s2-(p2-l2)**2
    return(np.round(U,4))




def P1_E_Payoffmatrix(adv):
    PM=np.zeros((num_actions, num_actions))
    for i in range(len(A)):
        for j in range(len(A)):
            
            
            s1=market_share(A[i],A[j],adv)
            PM[i,j]=U1_E(s1,A[i])

    return(PM)

def P2_E_Payoffmatrix(adv):
    PM=np.zeros((num_actions, num_actions))
    for i in range(len(A)):
        for j in range(len(A)):
            
            
            s2=1-market_share(A[i],A[j],adv)
            PM[i,j]=U2_E(s2,A[j])

    return(PM)
def U_G(s,e):
    k=1
    U=-0.5*(k/(1+s))*e**2
    return(np.round(U,4))
def G_Payoffmatrix1(s1): #directly a function of market  share of player 1 1 instead of state
    PM_G=np.zeros((len(E), len(E)))
    for i in range(len(E)):
        for j in range(len(E)):
            e=E[i]
            
            PM_G[i,j]=U_G(s1,e)
    return(PM_G)        

def G_Payoffmatrix2(s1):#directly a function of market share of player 2 1 instead of state
    PM_G=np.zeros((len(E), len(E)))
    for i in range(len(E)):
        for j in range(len(E)):
            e=E[j]
            
            s2=1-s1
            PM_G[i,j]=U_G(s2,e)
    return(PM_G)     
        


            


Shares.sort()
Game_stages=[0,1] #0 for G and 1 for E
ADV.sort()
GStates=Cross_product2([0],Shares)
EStates=Cross_product2([1],ADV)
Game_States=GStates+EStates

stateIDs = np.arange(len(Game_States))
Gamestate_dict = dict(zip(stateIDs, Game_States))

def get_state(stateID):
    return Gamestate_dict[stateID]


def get_stateID(state):
    
    for s, state_ in Gamestate_dict.items():
        
        if state_ == state:
            return s
    return None    


def Payoff_Matrix_1(g):#payoff matrix for player 1 for every game state
    Game_State=g[0]
    if Game_State==0:#if electoral state is G
        s1=g[1]
        PM=G_Payoffmatrix1(s1)
    else:
        adv=g[1]
        PM=P1_E_Payoffmatrix(adv)
    return(PM)


def Payoff_Matrix_2(g):#payoff matrix for player 1 for every game state
    Game_State=g[0]
    if Game_State==0:
        s1=g[1]
        PM=G_Payoffmatrix2(s1)
    else:
        adv=g[1]
        PM=P2_E_Payoffmatrix(adv)
    return(PM)


def TransitionMatrix_Stage1(a1,g):#transition matrix for actions of player 1 in E
    adv=g[1]
    num_action=len(A)
    TM=np.zeros((num_actions, len(Game_States)))
    for j in range(len(A)):
        s1=market_share(a1,A[j],adv)
        ES_next=1
        state_next=[0,s1]#stage2
        next_id=get_stateID(state_next)
        TM[j,next_id]=1
    return(TM)


        
def TransitionMatrix_Stage2(e1):#transition to E stage depending on action of player 1#
    num_action=len(E)
    TM=np.zeros((num_action, len(Game_States)))
    for j in range(len(E)):
        adv_next_1=e1-E[j]+noise
        adv_next_2=e1-E[j]-noise
        state_next1=[1,adv_next_1]
        state_next2=[1,adv_next_2]
        ID1=get_stateID(state_next1)
        ID2=get_stateID(state_next2)
        TM[j,ID1]=0.5
        TM[j,ID2]=0.5
    return(TM)    


def Transition_Matrix_state(S_ID):
    g=get_state(S_ID)
    Matrices=[]
    if  g[0]==0:
        for i in range(len(E)):
            tm=TransitionMatrix_Stage2(E[i])
            Matrices.append(tm)
        
    
    elif g[0]==1:
        adv=g[1]
        for i in range(len(A)):
            tm=TransitionMatrix_Stage1(A[i],g)
            Matrices.append(tm)

    Matrices=np.asarray(Matrices)
    return(Matrices)


FinaltransitionMatrices = [Transition_Matrix_state(s) for s in range(len(Game_States))]

def Payoff_Matrix_state(S_ID):
    g=get_state(S_ID)
    P_state=[]
    
    firm1_payoffmatrix=Payoff_Matrix_1(g)
    P_state.append(firm1_payoffmatrix)
    firm2_payoffmatrix=Payoff_Matrix_2(g)
    P_state.append(firm2_payoffmatrix)
    P_state=np.asarray(P_state)
    return(P_state)

FinalpayoffMatrices = [Payoff_Matrix_state(s) for s in range(len(Game_States))]

equilibrium = dsSolve(FinalpayoffMatrices, FinaltransitionMatrices, discountFactors=0.95, implementationType='ct', showProgress=True, plotPath=True)
