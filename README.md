# causal-gym
causal-gym

## Environment 1: Vacuum World

- Vacuum-v0
  - [dust, Agent]
  - frontdoor, $X \rightarrow W \rightarrow Y$, $X \leftarrow U \rightarrow Y$
- Expert:
  - Observation space $U$
    - 0: no dust, 1: dust
  - $f_{x} = U$
- Agent:
  - Observation space:
    - None
  - $\pi(x)$
- Action space $X$:
  - 0: do not move
  - 1: go left
- Realized action $W$:
  - the definition is the same as $X$
  - e.g., $P(W=1 \mid X=1) = 0.1$, $P(W=0 \mid X=0) = 0.1$
- Reward $Y$:
  - $Y=U \wedge W$
  - or $Y = U \oplus W$

$$
\begin{aligned}
    & P(U=0) = P(U=1) = 0.5 \\
    & P(X=0) = P(X=1) = 0.5 \\
    & P(X=1 \mid U=1) = 1.
\end{aligned}
$$

$$
\begin{aligned}
    P(Y=1) & = P(U=1, W=1) \\
    & = P(X=1, W=1) \\
    & = P(X=1) P(W=1 \mid X=1) \\
    & = 0.1*0.5 \\
    & = 0.05
\end{aligned}
$$

Frontdoor $\pi(x)$:
$$
\begin{aligned}
E[Y=1 \mid do(\pi)] & = \sum_{W} P(W | X) \sum_{X'} P(Y=1 \mid X', W) \times P(X') \\
& = \sum_{W} P(W | X) \left( P(Y=1 \mid X'=1, W=1) \times P(X'=1) \right) \\
\end{aligned}
$$


## Important Questions

1. How many environments?
  - Expert environment
  - Imitator environment
