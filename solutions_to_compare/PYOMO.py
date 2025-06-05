def solve_microgrid_pyomo(setup:MicrogridSetup):
    home_demands = []
    for _ in range(setup.no_homes):
        home_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1})
        home_demand = home_demand_dict["gen_consumption"]["consumption_kWh"]
        home_demands.append(home_demand)
    school_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100})
    school_demand = school_demand_dict["gen_consumption"]["consumption_kWh"]
    total_demand = [sum(group) for group in zip(*home_demands)] + school_demand

    wind_data_dict = get_wind_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    wind_prod_dict = generate_wind_turbine_data(wind_data_dict, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    wind_prod = wind_prod_dict["gen_wind_data"]["energy_generated"]

    solar_data_dict = get_irradiation_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    solar_prod_school_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": setup.solar_capacity_school})
    solar_prod_school = solar_prod_school_dict["gen_solar_data"]["energy_generated"]

    solar_prods = []
    for _ in range(setup.no_solar_panels):
        solar_data_dict =  get_irradiation_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
        solar_prod_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": setup.solar_capacity_home,})
        solar_prod = solar_prod_dict["gen_solar_data"]["energy_generated"]
        solar_prods.append(solar_prod)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, setup.T-1)
    model.B = pyo.RangeSet(0, setup.no_batteries-1)
    model.H = pyo.RangeSet(0, setup.no_homes-1)
    model.S = pyo.RangeSet(0, setup.no_big_solar_panels-1)

    model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.battery_charge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_discharge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_soc = pyo.Var(model.B, range(setup.T+1), domain=pyo.NonNegativeReals)
    model.c_home = pyo.Var(model.H, model.T, domain=pyo.NonNegativeReals)
    model.c_school = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    def soc_init_rule(m, b):
        return m.battery_soc[b,0] == 0
    model.soc_init = pyo.Constraint(model.B, rule=soc_init_rule)

    def power_balance_rule(m, t):
        total_hourly_battery_discharge = sum(m.battery_discharge[b, t] for b in model.B)
        total_hourly_battery_charge = sum(m.battery_charge[b, t] for b in model.B)
        total_hourly_solar_prod_homes = sum(m.c_home[h, t] * solar_prods[h][t] for h in model.H)
        total_hourly_solar_prod_school = m.c_school[t] * solar_prod_school[t]
        return (m.grid_import[t] + total_hourly_solar_prod_homes + total_hourly_solar_prod_school + wind_prod[t] + total_hourly_battery_discharge == 
                total_hourly_battery_charge + total_demand[t])
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    def battery_charge_limit_rule(m, b, t):
        return m.battery_charge[b, t] <= setup.battery_power
    model.battery_charge_limit = pyo.Constraint(model.B, model.T, rule=battery_charge_limit_rule)

    def battery_discharge_limit_rule(m, b, t):
        return m.battery_discharge[b, t] <= setup.battery_power
    model.battery_discharge_limit = pyo.Constraint(model.B, model.T, rule=battery_discharge_limit_rule)

    def soc_update_rule(m, b, t):
        return m.battery_soc[b, t+1] == m.battery_soc[b, t] + m.battery_charge[b, t] * setup.battery_efficiency - m.battery_discharge[b, t] / setup.battery_efficiency
    model.soc_update = pyo.Constraint(model.B, range(setup.T), rule=soc_update_rule)

    def soc_max_rule(m, b, t):
        return m.battery_soc[b, t+1] <= setup.battery_capacity
    model.soc_max = pyo.Constraint(model.B, range(setup.T), rule=soc_max_rule)

    def c_home_max_rule(m, h, t):
        return m.c_home[h, t] <= 1
    model.c_home_max = pyo.Constraint(model.H, range(setup.T), rule=c_home_max_rule)

    def c_school_max_rule(m, t):
        return m.c_school[t] <= 1
    model.c_school_max = pyo.Constraint(range(setup.T), rule=c_school_max_rule)

    model.obj = pyo.Objective(expr=sum(model.grid_import[t] for t in model.T), sense=pyo.minimize)

    solver = pyo.SolverFactory('cbc', executable="examples/Pyomo/cbc/bin/cbc.exe")
    solver.solve(model, tee=False)