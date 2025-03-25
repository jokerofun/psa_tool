# The Selector class can filter objects and extract attribute values.
class Selector:
    def __init__(self, items):
        self.items = items

    def of_type(self, type_or_types):
        """
        Filter items by instance type. Accepts a single type or a list/tuple of types.
        If a list/tuple is provided, the filter will include items that are instances of any of the types.
        """
        if isinstance(type_or_types, (list, tuple)):
            self.items = [item for item in self.items if any(isinstance(item, t) for t in type_or_types)]
        else:
            self.items = [item for item in self.items if isinstance(item, type_or_types)]
        return self

    def where(self, attr, value):
        """
        Filter items based on the value of an attribute.
        If 'value' is callable, it is treated as a comparator lambda and applied to the attribute value.
        Otherwise, an equality check is performed.
        """
        if callable(value):
            self.items = [item for item in self.items if value(item.get_attr(attr))]
        else:
            self.items = [item for item in self.items if item.get_attr(attr) == value]
        return self

    def values(self, attr):
        """
        Filters list of attribute values for each filtered item,
        using the generic get_attr method.
        """
        self.items = [item.get_attr(attr) for item in self.items]
        
    def get(self):
        """
        Returns the filtered items.
        """
        return self.items


# Example usage:
if __name__ == '__main__':
    # Example 1: Use Selector to filter for Foo instances and extract their costs.
    foo_costs = Selector(big_list).of_type(Foo).values("cost")
    print("Costs from Foo instances:", foo_costs)
    # goal is to use solve().maximize().of_type(Battery).values("cost")
    
    # Example 2: Use Selector to filter nodes with nodeID equal to 3.
    nodes_with_nodeID_3 = Selector(big_list).where("nodeID", 3).get()
    print("Nodes with NODEID equal to 3:", nodes_with_nodeID_3)

    # Example 3: Use Selector with a list of types (Foo and Bar) to extract all costs.
    all_costs = Selector(big_list).of_type([Foo, Bar]).values("cost")
    print("Costs from all Foo and Bar instances:", all_costs)
