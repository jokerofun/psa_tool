def solve_microgrid_gpod(setup: MicrogridSetup):
    problemClass = GraphProblemClass("microgrid_problem", time_length=setup.T)

    metering_point = MeteringPoint("grid1")
    homes = [Consumer(f"household_{i+1}") for i in range(setup.no_homes)]
    school = Consumer("school")
    solar_panels_homes = [SolarPanel(f"solar_panel_{i+1}") for i in range(setup.no_homes)]
    solar_panel_school = SolarPanel("solar_panel_school")
    wind_turbine = WindTurbine("wind_turbine")
    batteries = [Battery(f"battery_{i+1}", setup.battery_power, setup.battery_power, setup.battery_capacity, setup.battery_efficiency) for i in range(setup.no_batteries)]
    
    for home in homes:
        home.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1}, final=True)
    school.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100}, final=True)
    for solar_panel in solar_panels_homes:
        task1 = solar_panel.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
        task2 = solar_panel.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_home}, final=True)
        task1 >> task2
    solar_school_task1 = solar_panel_school.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217}) 
    solar_school_task2 = solar_panel_school.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_school}, final=True)
    solar_school_task1 >> solar_school_task2
    wind_task1 = wind_turbine.dataflow.task("get_wind_data", DataProcessingTask, process_func=get_wind_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
    wind_task2 = wind_turbine.dataflow.task("gen_wind_data", DataProcessingTask, process_func=generate_wind_turbine_data, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed": 14, "cut_out_speed": 25}, final=True)
    wind_task1 >> wind_task2
    
    problemClass.add_nodes([*homes, *solar_panels_homes, *batteries, 
                            school, solar_panel_school, wind_turbine, 
                            metering_point])
    
    metering_point.connect_to([*homes, *solar_panels_homes, *batteries, school, 
                                  solar_panel_school, wind_turbine])

    problemClass.solve(solver=cp.CBC, objective="minimize", value="cost")