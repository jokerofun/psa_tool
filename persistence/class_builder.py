import importlib
import os
import inspect

class ClassBuilder:
    def __init__(self, search_dirs):
         self.search_dirs = search_dirs

    def get_class_from_module(self, module_name, class_name):
        """Dynamically import a module and retrieve the class if it exists."""
        try:
            module = importlib.import_module(module_name)  # Dynamically import the module
            cls = getattr(module, class_name, None)  # Get the class
            if cls is None:
                raise ImportError(f"Class '{class_name}' not found in module '{module_name}'")
            return cls
        except ModuleNotFoundError:
            raise ImportError(f"Module '{module_name}' not found.")
    
    def instantiate_class(self, module_name, class_name, params):
        """Instantiate the class with provided parameters if it exists."""
        cls = self.get_class_from_module(module_name, class_name)
        
        # Get class constructor parameters
        constructor_params = inspect.signature(cls).parameters
        
        # Filter parameters that match constructor arguments
        valid_params = {k: v for k, v in params.items() if k in constructor_params}

        return cls(**valid_params)

    def find_class_file(self, class_name):
        """Search for a file where the class is defined in the given directories."""
        for directory in self.search_dirs:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".py"):  # Only Python files
                        module_name = file[:-3]  # Remove .py extension
                        module_path = os.path.join(root, file)
                        with open(module_path, "r", encoding="utf-8") as f:
                            if f"class {class_name}" in f.read():  # Check if class exists in the file
                                return module_name, root.replace("/", ".")  # Convert path to module format
        return None, None
    
    def build(self, class_name, params):
        # Locate the class file
        module_name, module_path = self.find_class_file(class_name)
        if module_name:
            full_module_name = f"{module_path}.{module_name}" if module_path else module_name
            obj = self.instantiate_class(full_module_name, class_name, params)
            print(obj)
            return obj
        else:
            print(f"Class '{class_name}' not found in search directories.")
