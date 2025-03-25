CREATE SEQUENCE IF NOT EXISTS parameter_id_seq;
CREATE TABLE IF NOT EXISTS parameters (
    parameter_id INTEGER DEFAULT nextval('parameter_id_seq') PRIMARY KEY,
    parameters_json JSON NULL
);

CREATE SEQUENCE IF NOT EXISTS node_id_seq;
CREATE TABLE IF NOT EXISTS nodes (
    node_id INTEGER DEFAULT nextval('node_id_seq') PRIMARY KEY,
    name TEXT NOT NULL,
    -- TODO: enum?
    class_type TEXT NOT NULL,
    parameter_id INTEGER NOT NULL,
    -- parent_id INTEGER,
    FOREIGN KEY (parameter_id) REFERENCES parameters(parameter_id)
);