
from enum import Enum
class Test:
    params = (
        ('TEST', 'paste')
    )
    def __init__(self, style = None) -> None:
        self.style = style
        Test.TEST = 'books'
        print(self.__getattribute__("TEST"))

    
    class Direction:
        @classmethod
        @property
        def Long(cls):
            return 'long'
        
        @classmethod
        @property
        def Short(cls):
            return 'short'
    

xxx = Test()
