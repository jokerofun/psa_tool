import duckdb
import json

class DBManager:
    def __init__(self, db_path="database.db"):
        self.conn = duckdb.connect(db_path)
        self.create_tables()

    def create_tables(self):
        with open('persistence/migrations/001-initial.sql', 'r') as file:
            sql_script = file.read()

        statements = sql_script.split(';')
        for stmt in statements:
            if stmt.strip() != '':
                self.conn.execute(stmt)

    def save_object(self, name, class_type, class_parameters):

        if self.load_object(name) is not None:
            return

        parameters_json = json.dumps(class_parameters)
        parameters_query = 'INSERT INTO parameters (parameters_json) ' \
                           'VALUES (?) ' \
                           'RETURNING parameter_id' 
        parameter_id = self.conn.execute(parameters_query, [parameters_json]).fetchone()[0]
        node_query = 'INSERT INTO nodes (name, class_type, parameter_id) ' \
                     'VALUES (?, ?, ?) ' \
                     'RETURNING node_id'
        node_query_parameters = [name, class_type, parameter_id]
        node_id = self.conn.execute(node_query, node_query_parameters).fetchone()[0]

        # return node_id

    def load_object(self, name):
        query = 'SELECT node_id, name, class_type, parameters_json FROM nodes ' \
                'JOIN parameters ON nodes.parameter_id = parameters.parameter_id ' \
                'WHERE nodes.name = ?'
        parameters = [name]

        result = self.conn.execute(query, parameters).fetchone()
        
        return result

    def get_connection(self):
        return self.conn

    def get_cursor(self):
        return self.get_connection().cursor()

    def close(self):
        self.get_connection().close()

    def execute(self, query, params=None):
        cursor = self.get_cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        self.get_connection().commit()
        cursor.close()
