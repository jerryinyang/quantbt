from abc import ABC, abstractmethod

class Observer(ABC):
    def __init__(self, name:str) -> None:
        self.name = name

    @abstractmethod
    def update(self, value):
        pass
