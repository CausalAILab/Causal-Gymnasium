class Graph:
    def __call__(self, observed_vars, unobserved_vars, edges):
        """Initialize the graph with observed and unobserved variables and edges."""
        self.observed_vars = observed_vars
        self.unobserved_vars = unobserved_vars
        self.edges = edges

    def get_graph(self):
        """Get the causal graph of the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_nodes(self):
        """Get all nodes in the graph."""
        return {"observed": self.get_observed_nodes(),
                "unobserved": self.get_unobserved_nodes()}
    
    def get_edges(self):
        """Get all edges in the graph."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_observed_nodes(self):
        """Get the observed nodes in the graph."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_unobserved_nodes(self):
        """Get the unobserved nodes in the graph."""
        raise NotImplementedError("This method should be implemented by subclasses.")