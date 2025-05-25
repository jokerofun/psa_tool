from .dataflow_classes_v2 import DataflowTask, DataFetchingFromFileTask, DataProcessingTask, MLTask
from typing import List, TypeVar
from src.dataflow.default_tasks import *

class Dataflow:
    def __init__(self, name, object) -> None:
        self.name = name
        self.object = object
        self.tasks = {}
        self.results = {}
        
    # also include optional arguments for the constructor    
    def task(self, name: str, task_node_type: DataflowTask = None, *args, **kwargs) -> DataflowTask:
        if name in self.tasks:
            return self.tasks[name]
        else:
            if task_node_type is None:
                self.tasks[name] = DataProcessingTask(name, *args, **kwargs)
            else:
                self.tasks[name] = task_node_type(name, *args, **kwargs)
            return self.tasks[name]
    
    def execute(self) -> None:
        print(f"Executing dataflow: {self.name}")
        final_task = self.get_final_task()
        if final_task is not None:
            final_task.run()
            self.results = final_task.get_results()

    # overload [] operator
    def __getitem__(self, name: str) -> DataflowTask:
        return self.task(name)
    
    def get_data(self, task_name):
        if task_name not in self.tasks:
            return {}
        
        task = self.tasks[task_name]
        return task.get_results()  # Ensure the task is run to get results
    
    def get_final_task(self) -> DataflowTask:
        """
        Get a final task in dataflow.
        """
        for task in self.tasks.values():
            if task._final:
                return task
        
        return None
    
    def get_tasks(self, names) -> List[DataflowTask]:
        """
        Get a list of tasks by their names.
        """
        return [self.tasks[name] for name in names if name in self.tasks]
    
    def default_workflow(self, 
                         source: str, 
                         select_func = None, 
                         cleanup_func = None,
                         format_func = None,
                         ml_func = None):
        """
        Create a default workflow with a data fetching, processing and ML tasks.
        """
        if not (source is None or source.strip() == ""):
            fetch_data_task = self.task("fetch_data", DataFetchingFromFileTask, source)

            if select_func is not None:
                select_data_task = self.task("select_data", DataProcessingTask, process_func=select_func, final=False)
            else:
                select_data_task = self.task("select_data", DataProcessingTask, process_func=do_nothing, final=False)
            fetch_data_task >> select_data_task

            if cleanup_func is not None:
                cleanup_data_task = self.task("cleanup_data", DataProcessingTask, process_func=cleanup_func, final=False)
            else:
                cleanup_data_task = self.task("cleanup_data", DataProcessingTask, process_func=do_nothing, final=False)
            select_data_task >> cleanup_data_task

            if format_func is not None:
                format_data_task = self.task("format_data", DataProcessingTask, process_func=format_func, final=False)
            else:
                format_data_task = self.task("format_data", DataProcessingTask, process_func=do_nothing, final=False)
            cleanup_data_task >> format_data_task

            if ml_func is not None:
                ml_task = self.task("ml_task", MLTask, model_func=ml_func, final=True)
            else:
                ml_task = self.task("ml_task", MLTask, model_func=do_nothing, final=True)
            format_data_task >> ml_task