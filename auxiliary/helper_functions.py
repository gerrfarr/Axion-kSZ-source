

def is_array(item, cl=None):
    try:
        if len(item)>1 and (cl is None or isinstance(item, cl)):
            return True
        else:
            return True
    except TypeError as ex:
        return False
    except AttributeError as ex:
        return False