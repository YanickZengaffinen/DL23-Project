from typing import Tuple

import numpy as np
import torch
import learn2learn as l2l

class TestDataset():
    def __init__(self, name: str, ways: int, shots: int, num_tasks: int) -> None:
        super().__init__()

        self.ways = ways
        self.shots = shots
        self.num_tasks = num_tasks

        self.tasksets = l2l.vision.benchmarks.get_tasksets(
            name,
            train_ways=ways,
            train_samples=2*shots,
            test_ways=ways,
            test_samples=2*shots,
            num_tasks=num_tasks,
            root='~/data',
        )
    
    def sample(self) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        data,labels = self.tasksets.test.sample()
        support_indices = np.zeros(data.size(0), dtype=bool)
        support_indices[np.arange(self.shots*self.ways) * 2] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support_x, support_y = data[support_indices], labels[support_indices]
        query_x, query_y = data[query_indices], labels[query_indices]

        return (support_x, support_y), (query_x, query_y)

