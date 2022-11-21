from typing import List

class IndicesTracker:
    def __init__(self) -> None:
        self.curr_indices = []
    
    def _is_empty(self):
        return self.curr_indices == []
    
    def reset(self) -> None:
        self.curr_indices = []

    def get_indices(self) -> List[int]:
        indices = self.curr_indices.pop(0)
        return indices
    
    def update(self, indices: List[int]) -> None:
        self.curr_indices.append(indices)
    
    def sanity_check(self):
        return self._is_empty()
    
    def is_last_batch(self):
        return self._is_empty()
