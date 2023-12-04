import re

from util import *

def extract_table_size(user_input):
    """ Use re to find how many people will be dining.
        :param str input: users input of booking info
        :return integer if found, None if not"""

    # TODO: could take the whole matched expression, tag the words and ensure they have the right context
    # Clarifies by NOT matching numbers followed by am or pm
    size_pattern = r'\b(table|party) for (\d+)(?!\s+[ap]m)\b'
    size_match = re.search(size_pattern, user_input)
    print(size_match)
    if size_match:
        return int(size_match.group(2))
    else:
        return None


def extract_time(user_input):
    # picks up the first value given
    # new vectorizer for booking intents: necessary? Or just look for times and 'table for'
    # TODO: Extract time (not 24h compatible. maybe look for at|for before hand?) Add the 24hour version at end
    time_pattern = r'\b(0?[1-9]|1[0-2])(:[0-5][0-9])? ?([ap]m)?\b'

    time_match = re.search(time_pattern, user_input, re.IGNORECASE)
    if time_match:
        # Return the info in the second brackets (?:(?:am|pm)
        booking_time = time_match.group(0)
        print(booking_time)
        return booking_time
    else:
        return None


print("%i people." % extract_table_size("I want to book a table for 4 at 10pm"))
print("At %s time." % extract_time("I want to book a table for 4 at 10pm"))
