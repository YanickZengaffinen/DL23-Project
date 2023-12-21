""" Standardized benchmark datasets that do a consistent train/val/test split """

from typing import Tuple
from learn2learn.vision.benchmarks import BenchmarkTasksets

import numpy as np
import torch
import torch.nn.functional as F
import learn2learn as l2l

class TasksetWrapper():
    def __init__(self, taskset, ways: int, shots: int, num_tasks: int) -> None:
        self.taskset = taskset
        self.ways = ways
        self.shots = shots
        self.num_tasks = num_tasks

    def sample(self) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        data,labels = self.taskset.sample()

        support_indices = np.zeros(data.size(0), dtype=bool)
        support_indices[np.arange(self.shots*self.ways) * 2] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support_x, support_lbls = data[support_indices], labels[support_indices]
        query_x, query_lbls = data[query_indices], labels[query_indices]

        return (support_x, support_lbls), (query_x, query_lbls)

def get_benchmark_tasksets(name: str, ways: int, shots: int, num_tasks: int) -> BenchmarkTasksets:
    tasksets = l2l.vision.benchmarks.get_tasksets(
                    name,
                    train_ways=ways,
                    train_samples=2*shots,
                    test_ways=ways,
                    test_samples=2*shots,
                    num_tasks=num_tasks,
                    root='~/data',
                )
    
    return BenchmarkTasksets(
        TasksetWrapper(tasksets.train, ways, shots, num_tasks), 
        TasksetWrapper(tasksets.validation, ways, shots, num_tasks), 
        TasksetWrapper(tasksets.test, ways, shots, num_tasks)
    )