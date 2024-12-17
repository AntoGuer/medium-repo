"""In this script, we will enrich a simple custom class with some dunder methods to better represent its instances"""


class Item:
    """
    A class representing an arbitrary object on sell, which is characterized by a name and a price.

    Attributes:
        name: The name of the Item instance
        price: The price pf the Item instance
    """

    def __init__(self, name: str, price: float) -> None:
        """
        Initialize object attribute 'name' and 'price'

        :param name: A string representing the name of the Item
        :param price: A float representing the unit price of the Item
        :return: None
        """
        self.name = name
        self.price = price

    def __repr__(self) -> str:
        """
        This method return a string object which will be displayed when trying to call on a console an object instance.
        To follow best practices, the output should provide the necessary syntax to recreate the printed instance.

        :return: A technical string representation of the object instance
        """
        return f"{self.__class__.__name__}('{self.name}', {self.price})"

    def __str__(self) -> str:
        """
        This method return a string object which will be displayed when trying to print an object instance.
        The output should provide a user-friendly representation of the object instance.

        NB: When attempting a print of an object instance, if this method is not defined the method __repr__ would be
        used.

        :return: A user-friendly string representation of the object instance to show
        """
        return f"{self.name}, which costs {self.price}$"


# Let's create an Item instance with arbitraries name and price
milk = Item(name="Milk (1L)", price=0.99)

# Let's display on Python console the variable item. We expect to have a technical representation of it (__repr__)
milk  # Item('Milk (1L)', 0.99)

# Let's print the variable item. We expect to have a user-friendly representation of it (__str__)
print(milk)  # Milk (1L), which costs 0.99$


