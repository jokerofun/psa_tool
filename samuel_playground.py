import pandas as pd
import duckdb
from persistence.class_builder import ClassBuilder
from persistence.db_manager import DBManager
from optimization.solver_classes import Battery, GraphProblemClass
import json

con = duckdb.connect('database.db')

db_context = DBManager()
problemClass = GraphProblemClass()
db_context.save_optimization_problem('problem1', 'problem description')
battery1 = Battery(problemClass, 50, 50, 'battery_1', 100)
battery2 = Battery(problemClass, 100, 100, 'battery_2', 200)

# print(battery1.get_class_info())
# print(battery1.get_parameters())

db_context.save_node(
    battery1.name,
    battery1.get_class_info(),
    battery1.get_parameters()
)
db_context.save_node(
    battery2.name,
    battery2.get_class_info(),
    battery2.get_parameters()
)
db_context.connect_problem_node('problem1', 'battery_1')
db_context.connect_problem_node('problem1', 'battery_2')

search_dirs = ['optimization']
class_builder = ClassBuilder(search_dirs)
data = db_context.load_node('battery_1')
print(data)
json_data = json.loads(data[3])
del json_data['connecting_node']
battery3 = class_builder.build(data[2], json_data)
# if data[2] == 'Battery':
#     battery3 = Battery(**json_data)
# data = { 
#     "problem_class": None, 
#     "name": "battery_3", 
#     "production_capacity": 150, 
#     "consumption_capacity": 150,
#     "efficiency": 0.9,
#     "battery_capacity": 200 }
# battery3 = Battery(**data)
# print(globals())

# result = con.execute('DROP TABLE nodes; DROP TABLE parameters; DROP SEQUENCE parameter_id_seq; DROP SEQUENCE node_id_seq;')
# print(result)
