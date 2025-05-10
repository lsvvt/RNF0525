import random
import time

from progress_table import ProgressTable

# Create table object:
table = ProgressTable(num_decimal_places=1)

# You can (optionally) define the columns at the beginning
table.add_column("x", color="bold red")

for step in range(10):
    x = random.randint(0, 200)

    # You can add entries in a compact way
    table["x"] = x

    # Or you can use the update method
    table.update("x", value=x, weight=1.0)

    # Display the progress bar by wrapping an iterator or an integer
    for _ in table(10):  # -> Equivalent to `table(range(10))`
        # Set and get values from the table
        table["y"] = random.randint(0, 200)
        table["x-y"] = table["x"] - table["y"]
        table.update("average x-y", value=table["x-y"], weight=1.0, aggregate="mean")
        time.sleep(0.1)

    # Go to the next row when you're ready
    table.next_row()

# Close the table when it's finished
table.close()
