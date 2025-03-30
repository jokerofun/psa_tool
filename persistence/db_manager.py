import duckdb
import json

from optimization.solver_classes import GraphProblemClass
from persistence.class_builder import ClassBuilder

class DBManager:
    def __init__(self, db_path="database.db", model_class_dirs=['optimization']):
        self.conn = duckdb.connect(db_path)
        self.__create_tables()
        self.class_builder = ClassBuilder(model_class_dirs)

    def __del__(self):
        self.close()

    def __create_tables(self):
        with open('persistence/migrations/001-initial.sql', 'r') as file:
            sql_script = file.read()

        statements = sql_script.split(';')
        for stmt in statements:
            if stmt.strip() != '':
                self.conn.execute(stmt)

    def save_optimization_problem(self, name, description):
        if self.load_optimization_problem(name) is not None:
            return
        
        query = 'INSERT INTO optimization_problems (name, description) ' \
                'VALUES (?, ?) ' \
                'RETURNING optimization_problem_id'
        parameters = [name, description]
        problem_class_id = self.conn.execute(query, parameters).fetchone()[0]
    
    def load_optimization_problem(self, name):
        query = 'SELECT optimization_problem_id, name, description ' \
                'FROM optimization_problems ' \
                'WHERE name = ?'
        parameters = [name]

        # TODO: fetch objects instead of tuples
        result = self.conn.execute(query, parameters).fetchdf()
        
        return result

    def load_optimization_problem_with_nodes(self, name):
        optimization_problem = self.load_optimization_problem(name)
        if optimization_problem.empty:
            return None
        
        query = 'SELECT nodes.node_id, name, class_type, parameters_json ' \
                'FROM nodes ' \
                'JOIN parameters ON nodes.parameter_id = parameters.parameter_id ' \
                'JOIN optimization_problems_nodes ON nodes.node_id = optimization_problems_nodes.node_id ' \
                'WHERE optimization_problems_nodes.optimization_problem_id = ?'
        parameters = [int(optimization_problem.loc[0, 'optimization_problem_id'])]

        nodes_df = self.conn.execute(query, parameters).fetchdf()
        print(nodes_df)
        if nodes_df.empty:
            return None
        
        problem_nodes = []
        for index, node in nodes_df.iterrows():
            node_params_json = json.loads(node['parameters_json'])
            del node_params_json['connecting_node']
            node_class = self.class_builder.build(node['class_type'], node_params_json)
            problem_nodes.append(node_class)

        problem = GraphProblemClass(name=name)
        problem.add_nodes(problem_nodes)

        return problem

    def save_node(self, node_name, class_type, class_parameters):

        if self.load_node_by_name(node_name) is not None:
            return

        parameters_json = json.dumps(class_parameters)
        parameters_query = 'INSERT INTO parameters (parameters_json) ' \
                           'VALUES (?) ' \
                           'RETURNING parameter_id' 
        parameter_id = self.conn.execute(parameters_query, [parameters_json]).fetchone()[0]
        node_query = 'INSERT INTO nodes (name, class_type, parameter_id) ' \
                     'VALUES (?, ?, ?) ' \
                     'RETURNING node_id'
        node_query_parameters = [node_name, class_type, parameter_id]
        node_id = self.conn.execute(node_query, node_query_parameters).fetchone()[0]

    def load_node_by_id(self, node_id):
        query = 'SELECT node_id, name, class_type, parameters_json FROM nodes ' \
                'JOIN parameters ON nodes.parameter_id = parameters.parameter_id ' \
                'WHERE nodes.node_id = ?'
        parameters = [node_id]

        result = self.conn.execute(query, parameters).fetchdf()

        return result
    
    def load_node_by_name(self, node_name):
        query = 'SELECT node_id, name, class_type, parameters_json FROM nodes ' \
                'JOIN parameters ON nodes.parameter_id = parameters.parameter_id ' \
                'WHERE nodes.name = ?'
        parameters = [node_name]

        result = self.conn.execute(query, parameters).fetchone()
        # print(result)
        return result

    def connect_problem_node(self, problem_name, node_name):
        problem = self.load_optimization_problem(problem_name)
        node = self.load_node_by_name(node_name)

        if problem is None or node is None:
            return
        
        problem_nodes = self.get_problem_nodes(problem_name)
        if node[0] in problem_nodes['node_id'].tolist():
            print("already connected")
            return
        
        query = 'INSERT INTO optimization_problems_nodes (optimization_problem_id, node_id) ' \
                'VALUES (?, ?)'
        parameters = [problem[0], node[0]]

        self.conn.execute(query, parameters)

    def get_problem_nodes(self, problem_name):
        problem = self.load_optimization_problem(problem_name)

        if problem is None:
            return
        
        query = 'SELECT node_id ' \
                'FROM optimization_problems_nodes ' \
                'WHERE optimization_problem_id = ?'
        parameters = [problem[0]]

        result = self.conn.execute(query, parameters).fetchdf()

        return result
    
    def disconnect_problem_node(self, problem_name, node_name):
        pass

    def get_connection(self):
        return self.conn

    def close(self):
        self.get_connection().close()

    def rebuild_database(self):
        self.conn.execute('DROP TABLE IF EXISTS optimization_problems_nodes')
        self.conn.execute('DROP TABLE IF EXISTS nodes')
        self.conn.execute('DROP TABLE IF EXISTS parameters')
        self.conn.execute('DROP SEQUENCE IF EXISTS parameter_id_seq')
        self.conn.execute('DROP SEQUENCE IF EXISTS node_id_seq')
        self.conn.execute('DROP TABLE IF EXISTS optimization_problems')
        self.conn.execute('DROP SEQUENCE IF EXISTS optimization_problem_id_seq')

        self.__create_tables()
