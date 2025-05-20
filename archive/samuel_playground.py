import pandas as pd
import duckdb
from persistence.class_builder import ClassBuilder
from persistence.db_manager import DBManager
from optimization.solver_classes import Battery, GraphProblemClass
import json

con = duckdb.connect('database.db')

db_context = DBManager()
# problemClass = GraphProblemClass()
# db_context.save_optimization_problem('problem1', 'problem description')
# battery1 = Battery(problemClass, 50, 50, 'battery_1', 100)
# battery2 = Battery(problemClass, 100, 100, 'battery_2', 200)

# # print(battery1.get_class_info())
# # print(battery1.get_parameters())

# db_context.save_node(
#     battery1.name,
#     battery1.get_class_info(),
#     battery1.get_parameters()
# )
# db_context.save_node(
#     battery2.name,
#     battery2.get_class_info(),
#     battery2.get_parameters()
# )
# db_context.connect_problem_node('problem1', 'battery_1')
# db_context.connect_problem_node('problem1', 'battery_2')

problem = db_context.load_optimization_problem_with_nodes('problem1')
print(problem)

# db_context.rebuild_database()
