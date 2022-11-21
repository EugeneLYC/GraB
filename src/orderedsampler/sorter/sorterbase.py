from typing import Dict, Union

class Sort:
    def __init__(self,
                prob_balance: bool = False,
                per_batch_order: bool = False) -> None:
        self.prob_balance = prob_balance
        self.per_batch_order = per_batch_order
    
    def reset_epoch(self):
        pass
    
    def step(self) -> Union[None, Dict]:
        raise NotImplementedError