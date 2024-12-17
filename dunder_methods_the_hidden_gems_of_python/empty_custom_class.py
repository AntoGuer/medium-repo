"""In this script, we will try to answer a simple question: is it possible to create an empty custom class?"""


# Let us define the "emptiest" custom class we can define in Python
class EmptyClass:
    """
    An empty class (no methods nor attributes)
    """
    pass


# You would maybe think that you can do nothing with it. But is it true?
empty_instance = EmptyClass()
another_empty_instance = EmptyClass()
print(empty_instance == another_empty_instance)  # False

# All these operations ran consistently. Is it magic? Let us deep dive the "content" of our EmptyClass
print(dir(EmptyClass))  # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', ...]

# See? The object is not empty at all: it has many methods (called 'dunder' that make possible to interact with it!)