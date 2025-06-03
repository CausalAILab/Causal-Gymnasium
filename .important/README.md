# Counterfactual UCB Reference Kit

*(Implements Algorithms 26 & 27 from **Bareinboim & Washington, 2024**)*

---

## 1  Folder layout

| file           | Textbook analogue                              | Purpose                                                                                                                                                                       |
| -------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ucbvi.py`** | **Algorithm 26 – Counterfactual UCB (driver)** | Maintains the episode loop, logs the triples *(state `s`, intent `x`, override `a`)* and their outcomes, **calls Step 7** to refresh optimism, then chooses actions greedily. |
| **`ucbq.py`**  | **Algorithm 27 – UCB‑Q sub‑routine**           | Pure function `ucb_q(...)` that **computes Step 7**: builds optimistic Q‑ and V‑tables from empirical counts, rewards, and the exploration bonus.                             |

Both files are entirely independent of Gym — pass primitive numpy arrays and call `act()` / `observe()` from your own environment wrapper.

---

## 2  How the code matches the book

### Exploration bonus

The bonus term used everywhere is
$\;b(s,x,a)=\;cH\,\sqrt{\tfrac{\ln(1/\delta)}{N(s,x,a)}}\;$
with **`c = 7`** (the constant from Azar‑Osband‑Munos, 2017).  This is precisely Eq. (9.47) preceding Alg 26.

### Algorithm 27 ➜ `ucbq.py`

* **Inputs** `C, R, H, δ` match the table in the margin of Alg 27.
* Line 4 in the book, *“for h = H … 1”*, is the backward loop in `ucb_q`.
* Line 6, *“Q = min{H, …}”*, is implemented with `np.minimum(H, …)`.
* The function returns **(Q, V)** exactly as required by Alg 26 Step 7.

### Algorithm 26 ➜ `ucbvi.py`

* **Step 3–4** (*peek intent; choose optimistic override*) → `act(s,x)`.
* **Step 5** (*execute do(a) and observe r,s′*) → handled by user’s environment, then fed into `observe(...)`.
* **Step 6** (update counts & rewards) → inside `observe`.
* **Step 7** (`Q ← UCB‑Q(…)`) → the call to `ucb_q` imported from `ucbq.py`.
* Remaining bookkeeping (episode reset, horizon handling) mirrors lines 1–2 & 8 in the text.

---

## 3  Minimal usage example

```python
from ucbvi import CounterfactualUCB
from frozen_lake_scm import FrozenLakeSCM   # any env exposing .action() & .do()

agent = CounterfactualUCB(S, A, H=100)

for episode in range(1000):
    agent.plan()                   # Step 7 (Alg 26)
    s = env.reset()
    done = False
    while not done:
        x_int = env.action()       # policy intent (peek only)
        a = agent.act(s, x_int)    # override picked via optimistic Q
        s2, r, done = env.do(a)
        agent.observe(s, x_int, a, r, s2)
        s = s2
```

Replace `FrozenLakeSCM` with any SCM‑style environment; only two calls are required:
`env.action()` (see intent) and `env.do(a)` (force override).

---

## 4  Tuning knobs

* **`H`** – horizon & reward cap (default: user‑supplied).
* **`c_bonus`** – exploration aggressiveness (default 7.0).
* **`delta`** – confidence level (default 0.05).

---

## 5  Limitations & next steps

* Dense numpy tensors are fine for toy problems; switch to sparse for |S|·|A|² ≫ 10⁶.
* Next‑intent distribution currently assumed uniform.  Predict or maximise if you need tighter bounds.
* No discount factor γ; finite‑horizon formulation follows the textbook.
