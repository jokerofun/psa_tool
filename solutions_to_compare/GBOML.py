def solve_microgrid_gboml(setup:MicrogridSetup, microgrid_file_path="examples/GBOML/microgrid.txt"):
    setup.T += 1
    gboml_domain.build_microgrid(setup=setup, file_path=microgrid_file_path)
    
    home_demands = []
    for _ in range(setup.no_homes):
        home_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1})
        home_demand = home_demand_dict["gen_consumption"]["consumption_kWh"]
        home_demands.append(home_demand)
    school_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100})
    school_demand = school_demand_dict["gen_consumption"]["consumption_kWh"]
    demand =  [sum(group) for group in zip(*home_demands)] + school_demand
    np.savetxt("data/gboml_data/demand.csv", demand)

    wind_data_dict = get_wind_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    wind_prod_dict = generate_wind_turbine_data(wind_data_dict, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    wind_prod = wind_prod_dict["gen_wind_data"]["energy_generated"]
    np.savetxt("data/gboml_data/gen_wind.csv", wind_prod[:setup.T])
    
    solar_data_dict = get_irradiation_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    solar_prod_school_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": setup.solar_capacity_school})
    solar_prod_school = solar_prod_school_dict["gen_solar_data"]["energy_generated"]
    np.savetxt("data/gboml_data/gen_big_solar.csv", solar_prod_school[:setup.T])

    solar_prod = {}
    for i in range(1, setup.no_solar_panels + 1):
        solar_data_dict = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})
        solar_prod_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": setup.solar_capacity_home,})
        solar_prod = solar_prod_dict["gen_solar_data"]["energy_generated"]
        np.savetxt(f"data/gboml_data/gen_solar_{i}.csv", solar_prod[:setup.T])

    gboml_model = GbomlGraph(setup.T)
    nodes, edges, global_params = gboml_model.import_all_nodes_and_edges(microgrid_file_path)
    gboml_model.add_nodes_in_model(*nodes)
    gboml_model.add_hyperedges_in_model(*edges)
    gboml_model.build_model()

    gboml_model.solve_clp()
    setup.T -= 1