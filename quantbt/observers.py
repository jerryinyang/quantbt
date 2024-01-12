from abc import ABC, abstractmethod


class Observer(ABC):
    def __init__(self, name:str) -> None:
        self.name = name

    @abstractmethod
    def update(self, value):
        pass

    # PICKLE-COMPATIBILITY
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Customize the object reconstruction
        self.__dict__.update(state)
