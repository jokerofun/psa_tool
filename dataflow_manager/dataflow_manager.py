from __future__ import annotations
import queue
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from dataflow_manager.dataflow_classes import DataflowNode

nodes_queue = queue.Queue()

def addNode(node: DataflowNode) -> None:
    nodes_queue.put(node)

def listQueue() -> None:
    print(list(nodes_queue.queue))

def addNodes(nodes: list[DataflowNode]) -> None:
    for node in nodes:
        nodes_queue.put(node)

def execute() -> Dict[str, pd.DataFrame]:
    dfs = {}
    while not nodes_queue.empty():
        node = nodes_queue.get()
        try:
            dfs = node.execute(dfs)
        except Exception as e:
            print(f"Error executing node {node.name}: {e}")
            nodes_queue.put(node)
            break
    return dfs