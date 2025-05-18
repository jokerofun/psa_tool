from gboml import GbomlGraph

gboml_model = GbomlGraph(24*365)
nodes, edges, _ = gboml_model.import_all_nodes_and_edges("experiments/GBOML/microgrid.txt")
gboml_model.add_nodes_in_model(*nodes)
gboml_model.add_hyperedges_in_model(*edges)
gboml_model.build_model()

solution = gboml_model.solve_clp()
print(solution)