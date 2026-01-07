"""
Test the get_variable_ordering utility function.

This test validates that the SCM.get_variable_ordering() method correctly
computes topological orderings based on causal graphs.
"""

import sys
import os

# Add parent directory to path  
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Direct file imports to bypass __init__.py
import importlib.util

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load required modules directly
types_path = os.path.join(parent_dir, 'causal_gym', 'core', 'types.py')
types_module = load_module_from_file('causal_gym.core.types', types_path)

scm_path = os.path.join(parent_dir, 'causal_gym', 'core', 'scm.py')
scm_module = load_module_from_file('causal_gym.core.scm', scm_path)

graph_defs_path = os.path.join(parent_dir, 'causal_gym', 'core', 'graph_defs.py')
graph_defs_module = load_module_from_file('causal_gym.core.graph_defs', graph_defs_path)

obj_utils_path = os.path.join(parent_dir, 'causal_gym', 'core', 'object_utils.py')
obj_utils_module = load_module_from_file('causal_gym.core.object_utils', obj_utils_path)

set_utils_path = os.path.join(parent_dir, 'causal_gym', 'core', 'set_utils.py')
set_utils_module = load_module_from_file('causal_gym.core.set_utils', set_utils_path)

graph_utils_path = os.path.join(parent_dir, 'causal_gym', 'core', 'graph_utils.py')
graph_utils_module = load_module_from_file('causal_gym.core.graph_utils', graph_utils_path)

graph_path = os.path.join(parent_dir, 'causal_gym', 'core', 'graph.py')
graph_module = load_module_from_file('causal_gym.core.graph', graph_path)

SCM = scm_module.SCM
Graph = graph_module.Graph


class SimpleTestSCM(SCM):
    """A minimal SCM for testing variable ordering."""
    
    def __init__(self):
        super().__init__()
        self.action_space = None
        self.observation_space = None
    
    def action(self):
        pass
    
    def observation(self):
        pass
    
    @property
    def get_graph(self):
        """Simple causal graph: U -> W <- D, W -> X"""
        nodes = [
            {'name': 'U', 'label': 'Exogenous U'},
            {'name': 'D', 'label': 'Exogenous D'},
            {'name': 'W', 'label': 'Warning'},
            {'name': 'X', 'label': 'Action'}
        ]
        edges = [
            {'from_': 'U', 'to_': 'W', 'type_': 'directed'},
            {'from_': 'D', 'to_': 'W', 'type_': 'directed'},
            {'from_': 'W', 'to_': 'X', 'type_': 'directed'}
        ]
        return Graph(nodes=nodes, edges=edges)


def test_variable_ordering_all_vars():
    """Test ordering all variables in the graph."""
    env = SimpleTestSCM()
    
    # Get ordering for all variables
    ordering = env.get_variable_ordering()
    
    print(f"Full variable ordering: {ordering}")
    
    # Check that U and D come before W
    u_idx = ordering.index('U')
    d_idx = ordering.index('D')
    w_idx = ordering.index('W')
    x_idx = ordering.index('X')
    
    assert u_idx < w_idx, f"U (index {u_idx}) should come before W (index {w_idx})"
    assert d_idx < w_idx, f"D (index {d_idx}) should come before W (index {w_idx})"
    assert w_idx < x_idx, f"W (index {w_idx}) should come before X (index {x_idx})"
    
    print("✓ All variables are in correct topological order")


def test_variable_ordering_subset():
    """Test ordering a subset of variables."""
    env = SimpleTestSCM()
    
    # Get ordering for a subset
    ordering = env.get_variable_ordering(['U', 'W'])
    
    print(f"Subset ordering (U, W): {ordering}")
    
    assert ordering == ['U', 'W'], f"Expected ['U', 'W'] but got {ordering}"
    
    print("✓ Subset ordering is correct")


def test_variable_ordering_missing_var():
    """Test that missing variables raise an error."""
    env = SimpleTestSCM()
    
    try:
        env.get_variable_ordering(['U', 'NONEXISTENT'])
        assert False, "Should have raised ValueError for missing variable"
    except ValueError as e:
        print(f"✓ Correctly raised error for missing variable: {e}")


def test_variable_ordering_independent_vars():
    """Test that independent variables (U and D) can be in any order."""
    env = SimpleTestSCM()
    
    ordering = env.get_variable_ordering(['U', 'D'])
    
    print(f"Independent variables ordering: {ordering}")
    
    # Both orderings are valid for independent variables
    assert set(ordering) == {'U', 'D'}, f"Expected U and D, got {ordering}"
    
    print("✓ Independent variables ordering is valid")


if __name__ == '__main__':
    print("Testing SCM.get_variable_ordering()...\n")
    
    test_variable_ordering_all_vars()
    print()
    
    test_variable_ordering_subset()
    print()
    
    test_variable_ordering_missing_var()
    print()
    
    test_variable_ordering_independent_vars()
    print()
    
    print("All tests passed! ✓")
