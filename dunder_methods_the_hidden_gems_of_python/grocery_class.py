"""In this script, we will enrich a custom class with some dunder methods to better interact with its instances"""

from dunder_methods_the_hidden_gems_of_python.item_class import Item
from typing import Optional, Iterator
from typing_extensions import Self


class Grocery:
    """
    A class representing a grocery list, where items and their quantities are stored.

    Attributes:
        items: The list of Item in the Grocery instance along with their quantities
    """

    def __init__(self, items: Optional[dict[Item, int]] = None) -> None:
        """
        Initializes object attribute 'items' with a dictionary associating items with their quantities.

        :param items: A dictionary associating Item objects with their quantities.
        :return: None
        """
        self.items = items or dict()

    def __add__(self, new_items: dict[Item, int]) -> Self:
        """
        Adds new items to the Grocery instance, combining quantities for duplicate items.

        :param new_items: A dictionary associating the Item objects to add to the instance along with their quantities
        :return: A new Grocery instance with the updated 'items' attribute
        """

        new_grocery = Grocery(items=self.items)

        for new_item, quantity in new_items.items():

            if new_item in new_grocery.items:
                new_grocery.items[new_item] += quantity
            else:
                new_grocery.items[new_item] = quantity

        return new_grocery

    def __iter__(self) -> Iterator[Item]:
        """
        Returns an iterator over the Grocery instance

        :return: An iterator object built over the 'items' attributes of the Grocery instance
        """
        return iter(self.items)

    def __getitem__(self, item: Item) -> int:
        """
        Retrieves the quantity of a specific item in the Grocery instance.
        Raises a KeyError if the Item is no present in the Grocery instance.

        :param item: An Item object possibly present in some quantity inside the 'items' attribute of the instance
        :return: An integer representing the quantity of Item in the 'items' attribute of the instance
        """

        if self.items.get(item):
            return self.items.get(item)
        else:
            raise KeyError(f"Item {item} not in the grocery")


# Let's create an Item object and a Grocery instance with that item in some quantity.
milk = Item(name="Milk (1L)", price=0.99)
grocery = Grocery(items={milk: 3})

# Let's print out the 'items' attribute of the 'grocery' instance
print(grocery.items)  # {Item('Milk (1L)', 0.99): 3}

# Let's add a new item in the grocery (__add__)
soy_sauce = Item(name="Soy Sauce (0.375L)", price=1.99)
grocery = grocery + {soy_sauce: 1} + {milk: 2}

# Let's print the updated 'grocery' instance
print(grocery.items)  # {Item('Milk (1L)', 0.99): 5, Item('Soy Sauce (0.375L)', 1.99): 1}

# Let's use a list-comprehension to iterate over the 'grocery' instance
print([item for item in grocery])  # [Item('Milk (1L)', 0.99), Item('Soy Sauce (0.375L)', 1.99)]

# Let's try to access to the quantity of 'new_item' in the 'grocery' instance
print(grocery[soy_sauce])  # 1

# Let's define a new item that is not in the 'grocery' instance and again let's try to access its quantity
fake_item = Item("Creamy Cheese (500g)", 2.99)
print(grocery[fake_item])  # KeyError: "Item Item('Creamy Cheese (500g)', 2.99) not in the grocery"