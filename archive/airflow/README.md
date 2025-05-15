# how to init airflow (in linux)

```airflow standalone``` 

Initialize the webserver by default the config file is in `~/airflow`

## add dags to your project
in the .cfg file add this
```
[core]
dags_folder = /home/sobibence/AAU/3_semester/project/psa_tool/airflow/dags
```