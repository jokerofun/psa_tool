def do_nothing(dfs):
    """A placeholder function that does nothing."""
    return dfs

def procFunc1(dfs):
    # do some processing
    print("Processing data")
    dfs = dfs["csv_prices"] 
    return dfs

def trainFunc1(dfs):
    # do some processing
    print("Training model")
    return dfs