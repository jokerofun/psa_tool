from optimization.solver_classes import GraphProblemClass
from optimization.energy_sector_classes import Battery, PowerExchange
from persistence.db_manager import DBManager

if __name__ == "__main__":
    db_context = DBManager()

    problemClass = GraphProblemClass("batteryArbitrage")
    battery1 = Battery(problemClass, 50, 50, "bat1", 100 )
    battery2 = Battery(problemClass, 100, 100,"bat2",  200)
    battery3 = Battery(problemClass, 150, 150, "bat3", 300)
    power_exchange = PowerExchange(problemClass, 50, 50, "powerExchange")

    # Connect the nodes
    power_exchange - battery1
    power_exchange - battery2
    power_exchange - battery3

    db_context.save_node(battery1.name, battery1.__class__.__name__, battery1.get_parameters())
    db_context.save_node(battery2.name, battery2.__class__.__name__, battery2.get_parameters())
    db_context.save_node(battery3.name, battery3.__class__.__name__, battery3.get_parameters())
    db_context.save_node(power_exchange.name, power_exchange.__class__.__name__, power_exchange.get_parameters())
    db_context.save_optimization_problem(problemClass.name, 'battery arbitrage example')
    db_context.connect_problem_node(problemClass.name, battery1.name)
    db_context.connect_problem_node(problemClass.name, battery2.name)
    db_context.connect_problem_node(problemClass.name, battery3.name)
    db_context.connect_problem_node(problemClass.name, power_exchange.name)