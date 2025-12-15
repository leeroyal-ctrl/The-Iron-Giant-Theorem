python
    """Base class for all DAA components"""
    def serialize(self):
        """Return model state as JSON-compatible dict"""
        raise NotImplementedError("Subclasses must implement serialize()")
    
    def visualize(self):
        """Generate explanatory visualization diagram path"""
        raise NotImplementedError("Visualization required for all models")
