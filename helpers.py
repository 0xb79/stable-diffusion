def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False