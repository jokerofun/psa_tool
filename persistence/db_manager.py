import duckdb
import json

class DBManager:
    def __init__(self, db_path="database.db"):
        self.conn = duckdb.connect(db_path)
        self.__create_tables()

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

        # TODO: fetch dfs
        result = self.conn.execute(query, parameters).fetchone()

        return result

    def load_optimization_problem_with_nodes(self, name):
        optimization_problem = self.load_optimization_problem(name)
        if optimization_problem is None:
            return None
        
        query = 'SELECT node_id, name, class_type, parameters_json ' \
                'FROM nodes ' \
                'JOIN optimization_problems_nodes ON nodes.node_id = optimization_problems_nodes.node_id ' \
                'WHERE optimization_problem_id = ?'
        parameters = [optimization_problem[0]]
        nodes = self.conn.execute(query, parameters).fetchall()

        return nodes

    def save_node(self, node_name, class_type, class_parameters):

        if self.load_node(node_name) is not None:
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

    def load_node(self, node_name):
        query = 'SELECT node_id, name, class_type, parameters_json FROM nodes ' \
                'JOIN parameters ON nodes.parameter_id = parameters.parameter_id ' \
                'WHERE nodes.name = ?'
        parameters = [node_name]

        result = self.conn.execute(query, parameters).fetchone()
        
        return result

    def get_connection(self):
        return self.conn

    def close(self):
        self.get_connection().close()
