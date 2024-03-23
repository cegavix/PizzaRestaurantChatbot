import re
import sqlite3
import nltk


def extract_table_size(user_input):
    """ Use re to find how many people will be dining.
        :param str input: users input of booking info
        :return integer if found, None if not"""
    tokens = nltk.word_tokenize(user_input)
    tagged_tokens = nltk.pos_tag(tokens)
    if len(tokens) == 1 and tagged_tokens[0][1] == 'CD':
        return user_input
    # TODO: could take the whole matched expression, tag the words and ensure they have the right context
    # Clarifies by NOT matching numbers followed by am or pm
    size_pattern = r'\b(table|party) for (\d+)(?!\s+[ap]m) {people?}\b'
    size_pattern2 = r'\b(for)?\s?(\d+)(?!\s+[ap]m)\s?(people)?\b'
    size_match = re.search(size_pattern, user_input, re.IGNORECASE)
    size_match2 = re.search(size_pattern2, user_input, re.IGNORECASE)
    if size_match:
        return int(size_match.group(2))
    elif size_match2:
        return int(size_match2.group(2))
    else:
        return None


def extract_time(user_input):
    """ Use re to look for the time described
    :param user_input: Presumes user is responding to prompt
    :return:
    """
    # TODO: Extract time (not 24h compatible. maybe look for at|for before hand?) Add the 24hour version at end
    time_pattern = r'\b(0?[1-9]|1[0-2])(:[0-5][0-9])?\s?([ap]m)?(?!\s?people)\b'
    time_matches = list(re.finditer(time_pattern, user_input, re.IGNORECASE))
    if time_matches:
        # Returns the longest match
        time_matches.sort(key=lambda match: len(match.group(0)), reverse=True)
        booking_time = time_matches[0].group(0)
        return booking_time
    else:
        return None


def insert_into_database(name, time, people_num):
    try:
        connection = sqlite3.connect('booking.db')
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS bookings
                        (name text NOT NULL, start_time text NOT NULL, table_size int NOT NULL);''')

        cursor.execute("INSERT INTO bookings VALUES (?,?,?)", (name, time, people_num))
        task1 = cursor.execute("SELECT * FROM bookings")

        connection.commit()
    except sqlite3.Error as error:
        # Handle the error
        print(f"Error: {error}")
        errored = True
    else:
        # print("Booking added to database:", task1.fetchall())
        errored = False

    finally:
        if connection:
            connection.close()

    return errored


def make_booking(user_input, name):
    """ Prompts user until they provide valid booking information. Inserts the data into the database.
    :param user_input: what the user said
    :param name: the users name
    """

    table_size = extract_table_size(user_input)
    booking_time = extract_time(user_input)

    while booking_time is None:
        print("Papa: What time would you like to book for?")
        user_input = input('%s: ' % name)
        booking_time = extract_time(user_input)
        if booking_time is None:
            print(
                "Papa: I'm sorry, I didn't quite catch the time for your booking. Try specifying the time of day with"
                " a 'am' or 'pm'.")

    while table_size is None:
        print("Papa: Bellisima! How many people would you like to book for?")
        user_input = input('%s: ' % name)
        table_size = extract_table_size(user_input)
        if table_size is None:
            print(
                "Papa: I'm sorry, I didn't quite catch the table size needed for your booking. Say a table for 2, "
                "for example.")
        elif table_size > 5:
            print("Papa: Wow, you're popular huh! Sounds like a party.")

    print(f"Papa: You would like to make a booking with these details:\n"
          f"   Time: {booking_time} \n"
          f"   Size: {table_size} people.\n "
          f"If these details are correct, please say yes.")
    confirmed = input('%s: ' % name).lower()

    if confirmed == 'yes':
        insert_into_database(name, booking_time, table_size)
        print("Papa: Great! Your booking has been made successfully.")
    else:
        print("Papa: My apologies, %s. Lets try this again. What time and table size would you like for your booking?" % name)
        change = input('%s: ' % name)
        make_booking(change, name)

