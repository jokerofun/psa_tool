CREATE SEQUENCE IF NOT EXISTS optimization_problem_id_seq;
CREATE TABLE IF NOT EXISTS optimization_problems (
    optimization_problem_id INTEGER DEFAULT nextval('optimization_problem_id_seq') PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NULL
);

CREATE SEQUENCE IF NOT EXISTS parameter_id_seq;
CREATE TABLE IF NOT EXISTS parameters (
    parameter_id INTEGER DEFAULT nextval('parameter_id_seq') PRIMARY KEY,
    parameters_json JSON NULL
);

CREATE SEQUENCE IF NOT EXISTS node_id_seq;
CREATE TABLE IF NOT EXISTS nodes (
    node_id INTEGER DEFAULT nextval('node_id_seq') PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    -- TODO: enum?
    class_type TEXT NOT NULL,
    parameter_id INTEGER NOT NULL,
    FOREIGN KEY (parameter_id) REFERENCES parameters(parameter_id)
);

CREATE TABLE IF NOT EXISTS optimization_problems_nodes (
    optimization_problem_id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    FOREIGN KEY (optimization_problem_id) REFERENCES optimization_problems(optimization_problem_id),
    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
);