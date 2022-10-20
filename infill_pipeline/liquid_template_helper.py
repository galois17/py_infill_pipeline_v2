
def filter_string_format(val, fmt):
    """ Liquid filter for formatting 
    
    Args:
        val: the value
        fmt: the format such as '%9.5f'
    Returns:
        the formatted string
    """
    return fmt % (val)