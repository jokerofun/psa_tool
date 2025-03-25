import pandas as pd
import duckdb
from persistence.db_manager import DBManager
from optimization.solver_classes import Battery, GraphProblemClass

con = duckdb.connect('database.db')

db_context = DBManager()

problemClass = GraphProblemClass()
battery1 = Battery(problemClass, 50, 50, 'battery_1', 100)
battery2 = Battery(problemClass, 100, 100, 'battery_2', 200)

# print(battery1.get_class_info())
# print(battery1.get_parameters())

db_context.save_object(
    battery1.name,
    battery1.get_class_info(),
    battery1.get_parameters()
)
db_context.save_object(
    battery2.name,
    battery2.get_class_info(),
    battery2.get_parameters()
)

battery3 = None
data = db_context.load_object('battery_1')
print(data[3])
if data[2] == 'Battery':
    battery3 = Battery(**data[3])
print(battery3)

parameters = con.execute('SELECT * FROM parameters;')
print(parameters.fetchdf())
nodes = con.execute('SELECT * FROM nodes;')
print(nodes.fetchdf())

# result = con.execute('DROP TABLE nodes; DROP TABLE parameters; DROP SEQUENCE parameter_id_seq; DROP SEQUENCE node_id_seq;')
# print(result)
