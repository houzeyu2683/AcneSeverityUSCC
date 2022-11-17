
def createUnit(name='unit'):

    unit = type(name, (type,), {})
    return(unit)

