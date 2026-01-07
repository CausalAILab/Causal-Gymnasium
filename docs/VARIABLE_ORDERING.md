# Variable Ordering in SCM Environments

## Overview

When implementing Structural Causal Model (SCM) environments in CausalGym, **the order in which variables are sampled or computed matters**. This is because endogenous variables often depend on other variables (their "parents" in the causal graph), and attempting to compute a variable before its parents are available will result in errors or incorrect behavior.

## The Problem

Consider a simple causal model with three variables:
- `U`: Weather condition (exogenous, sampled from a distribution)
- `D`: Driver impairment status (exogenous, sampled from a distribution)
- `W`: Dashboard warning (endogenous, computed as a function of U and D)

The causal graph looks like this:
```
U → W
D → W
```

### ❌ Incorrect Implementation

```python
def reset(self):
    # WRONG: Computing W before U and D are sampled
    self.W = self.calc_W(self.U, self.D)  # Error! U and D are not defined yet
    self.U = self.sample_U()
    self.D = self.sample_D()
```

### ✅ Correct Implementation

```python
def reset(self):
    # CORRECT: Sample/compute in topological order
    self.U = self.sample_U()  # Sample exogenous variables first
    self.D = self.sample_D()
    self.W = self.calc_W(self.U, self.D)  # Compute endogenous variables after their parents
```

## Why Does Ordering Matter?

In an SCM, each endogenous variable is defined by a structural equation:
```
V = f_V(Parents(V), U_V)
```

Where:
- `V` is the endogenous variable
- `Parents(V)` are the causal parents of V in the graph
- `U_V` is the exogenous noise for V
- `f_V` is the structural function

**To compute V, all variables in Parents(V) must already have values.**

## Using the `get_variable_ordering` Helper

CausalGym's base `SCM` class provides a `get_variable_ordering()` method that automatically computes the correct topological order based on your causal graph:

```python
from causal_gym import SCM

class MyEnvironment(SCM):
    def reset(self):
        # Get the correct ordering
        ordered_vars = self.get_variable_ordering(['U', 'D', 'W', 'X'])
        
        # Now sample/compute in the correct order
        for var in ordered_vars:
            if var == 'U':
                self.U = self.sample_U()
            elif var == 'D':
                self.D = self.sample_D()
            elif var == 'W':
                self.W = self.calc_W(self.U, self.D)
            elif var == 'X':
                self.X = self.calc_X(self.W)
        
        return self.observation(), {}
```

## Real-World Example: Race Environment

The `RaceSCM` environment in `causal_gym/envs/race.py` demonstrates proper variable ordering:

```python
def reset(self):
    # Step 1: Sample exogenous variables
    self._D = self.sample_D()  # Driver drunk/sober
    self._U = [self.sample_U()]  # Fog
    
    # Step 2: Compute endogenous variables that depend on exogenous ones
    self.W = [self.calc_W(self._D, self._U[self.t])]  # Warning (depends on D and U)
    
    # Step 3: Compute other endogenous variables
    self.C = [self.calc_C()]  # Lane centering
    self.H = [self.calc_H()]  # Heading error
```

The causal relationships are:
```
D → W
U → W
X → C (from previous timestep)
X → H, H → H (from previous timestep)
```

## Common Patterns

### Pattern 1: Independent Exogenous Variables
When variables have no dependencies between them, order doesn't matter:
```python
# These can be in any order
self.U1 = self.sample_U1()
self.U2 = self.sample_U2()
self.U3 = self.sample_U3()
```

### Pattern 2: Chain Dependencies
For chains like `A → B → C`:
```python
# Must be in this order
self.A = self.sample_A()
self.B = self.calc_B(self.A)
self.C = self.calc_C(self.B)
```

### Pattern 3: Multiple Parents
For `A → C ← B`:
```python
# A and B can be in any order, but both must come before C
self.A = self.sample_A()
self.B = self.sample_B()
self.C = self.calc_C(self.A, self.B)  # Depends on both A and B
```

## Temporal Dependencies

In sequential environments, variables from the previous timestep are already computed, so they can be used as parents:

```python
def step(self, action):
    self.t += 1
    
    # Variables from previous timestep (self.X[t-1], self.H[t-1]) are already available
    # Sample new exogenous variables for current timestep
    self._U.append(self.sample_U())
    
    # Compute new endogenous variables that may depend on previous timestep
    self.W.append(self.calc_W(self._D, self._U[self.t]))  # D is constant across time
    self.C.append(self.calc_C())  # May implicitly depend on X[t-1]
    self.H.append(self.calc_H())  # May depend on H[t-1] and X[t-1]
```

## Tips for Implementers

1. **Design your causal graph first**: Before implementing `reset()` or `step()`, clearly define the causal relationships using `get_graph`.

2. **Use `get_variable_ordering()` during development**: This helps catch ordering issues early:
   ```python
   ordered_vars = self.get_variable_ordering()
   print(f"Variables should be computed in this order: {ordered_vars}")
   ```

3. **Group variables by dependency level**: Organize your code by "layers":
   - Layer 0: Exogenous variables (no parents)
   - Layer 1: Variables that only depend on exogenous variables
   - Layer 2: Variables that depend on Layer 1 variables
   - And so on...

4. **Document dependencies in comments**: Help future maintainers understand the dependencies:
   ```python
   # Exogenous variables
   self.U = self.sample_U()
   self.D = self.sample_D()
   
   # Endogenous variables (depends on U and D)
   self.W = self.calc_W(self.U, self.D)
   ```

5. **Test with different orderings**: If you're unsure about dependencies, try swapping the order and see if your code breaks. If it doesn't break, the variables might not actually depend on each other.

## Technical Details

The `get_variable_ordering()` method uses topological sorting on the causal graph:
- It ensures that for every directed edge `U → V`, variable `U` appears before `V` in the ordering
- For variables with no path between them, the relative order is arbitrary (but deterministic when `sort_=True`)
- If the graph contains cycles, topological sorting will fail (as it should, since SCMs require acyclic graphs for identifiability)

## See Also

- Graph utilities: `causal_gym/core/graph_utils.py`
- Example implementations: `causal_gym/envs/race.py`, `causal_gym/envs/highway_single_step.py`
- Pearl's Causality framework for understanding SCMs
