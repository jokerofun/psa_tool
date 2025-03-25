import pandas as pd
import duckdb
from persistence.db_manager import DBManager
from optimization.solver_classes import Battery, GraphProblemClass
import json

con = duckdb.connect('database.db')

db_context = DBManager()
problemClass = GraphProblemClass()
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

battery3 = None
data = db_context.load_node('battery_1')
print(data[3])
json_data = json.loads(data[3])
del json_data['connecting_node']
print(type(json_data))
if data[2] == 'Battery':
    battery3 = Battery(**json_data)
# data = { 
#     "problem_class": None, 
#     "name": "battery_3", 
#     "production_capacity": 150, 
#     "consumption_capacity": 150,
#     "efficiency": 0.9,
#     "battery_capacity": 200 }
# print(data)
# battery3 = Battery(**data)
print(battery3)
print(globals())

# parameters = con.execute('SELECT * FROM parameters;')
# print(parameters.fetchdf())
# nodes = con.execute('SELECT * FROM nodes;')
# print(nodes.fetchdf())

# result = con.execute('DROP TABLE nodes; DROP TABLE parameters; DROP SEQUENCE parameter_id_seq; DROP SEQUENCE node_id_seq;')
# print(result)
