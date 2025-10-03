# Tullock-Contest-in-a-MARL-context
This is a project idea for Multi Agent Reinforcement Learning course which is part of my PhD training scheme. The Tullock Contest is more complex, in this set-up, only the simplest case is considered.  
## Environment description

1. Each episode = one Tullock contest; alternatively, one contest can be treated as a single step, repeated indefinitely.
   
2. Minimal observation (for each agent): previous effort 
$𝑥_i$, total investment A, and whether the agent won.

3. Decision: choose effort 
$x_i$

## Reward options：
    
    1. Expected payoff (smoother, more stable convergence): 

$$
r_i = p_i \cdot V - x_i
$$
    
    1. Stochastic payoff (lottery-style allocation, more realistic but noisier)：

$$
r_i = \mathbf{1}_{\{\text{win}\}} \cdot V - x_i
$$

## Contest Success Function (CSF)

$$p_i = \frac{a_i x_i^{r_i}}{\sum_j a_j x_j^{r_j} + \epsilon}$$

- $a_i$：production efficiency (default $a_i=1$)  
- $r_i$：elasticity parameter (default $r_i=1$) 
- $\epsilon$：smoothing term to avoid division by zero  

### Cost and reward functions
- Linear cost:

$$
c(x_i) = x_i
$$

- Expected reward (recommended for training stability):

$$
r_i = p_i \cdot V - x_i
$$

where $V$ is the prize value(default $V=1$)。

### Action space
 
  $$
  x_i \in [0, x_{\max}]
  $$
  
  default $x_{\max}=1.0$

- **Observation space (minimal)**：  
  - Previous effort $x_i$  
  - Total investment $A = \sum_j x_j$  

---

### Environment design
Based on `gymnasium` / `pettingzoo`：

```python
# reset(): returns obs = {agent_i: obs_i}
# step(actions): input {agent_i: action_i}，return (obs, rewards, dones, infos)
```
### WORKFLOW
```
marl-tullock/
├─ src/
│  ├─ envs/             
│  │   └─ tullock_env.py
│  ├─ training/         
│  │   └─ train_mappo.py
│  └─ utils/            
├─ tests/               
├─ scripts/             
├─ requirements.txt     
├─ README.md            
└─ .gitignore           
```

