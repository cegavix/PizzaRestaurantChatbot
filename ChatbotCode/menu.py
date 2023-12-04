menu = [
    {"Dish": "Margherita Pizza", "Description": "Classic pizza with tomato, fresh mozzarella, and basil.",
     "Price": "$10.99", "Dietary Requirements": "Vegetarian, Low sodium. Dairy Free options available."},
    {"Dish": "Spaghetti Bolognese", "Description": "Spaghetti pasta with a rich meat sauce.", "Price": "$12.99",
     "Dietary Requirements": "Vegetarian option available, Low sodium. Dairy Free options available. May Contain Nuts"},
    {"Dish": "Caprese Salad", "Description": "Fresh tomatoes, mozzarella, and basil drizzled with balsamic glaze.",
     "Price": "$8.99"},
    {"Dish": "Chicken Parmesan", "Description": "Breaded and fried chicken topped with marinara sauce and melted "
                                                "cheese.", "Price": "$14.99"},
    {"Dish": "Ravioli al Pomodoro", "Description": "Homemade ravioli stuffed with ricotta cheese, served with tomato "
                                                   "sauce.", "Price": "$11.99"},
    {"Dish": "Tiramisu", "Description": "Traditional Italian dessert made with layers of coffee-soaked ladyfingers "
                                        "and mascarpone cheese.", "Price": "$6.99"},
    {"Dish": "Espresso", "Description": "Strong and rich Italian coffee.", "Price": "$3.99"},
    {"Dish": "Gelato", "Description": "Assorted flavors of Italian ice cream.", "Price": "$5.99"}
]
def print_menu_entry(index):
    # Print the menu
    print("We have a range of delicious options on our Menu:")
    print("{:<25} {:<70} {:<10}".format("Dish", "Description", "Price"))
    print("-" * 105)
    item = menu[index]
    print("{:<25} {:<70} {:<10}".format(item["Dish"], item["Description"], item["Price"],
                                        item["Dietary Requirements"]))


def print_menu():
    # Print the menu
    print("Here is the dish that matches your query:")
    print("{:<25} {:<70} {:<10}".format("Dish", "Description", "Price"))
    print("-" * 105)

    for item in menu:
        print("{:<25} {:<70} {:<10}".format(item["Dish"], item["Description"], item["Price"]))