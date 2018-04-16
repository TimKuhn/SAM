import pandas as pd
import re
import os

def _loadCSV(filePath, sep, headerRow):
    return pd.read_csv(filePath, sep, header=headerRow)

def _readExcel(filePath, sep, headerRow):
    return pd.read_excel(filePath, header=headerRow)

def loadFile(filePath, sep=",", headerRow=None, columnIndex=None):
    """ Load a file

    :param filePath: Path to file
    :param sep: If csv file -> please specify separator. Comma by default
    :param headerRow: Integer that specifies the row where the header is located
    :param columnIndex: Integer specifying column in file (start with 0)

    :returns: returns a panda dataframe
    """

    # Get extension from filePath
    extension = filePath.rsplit(".")[-1]

    # Look-up table to call specific load function based on fileType
    validFileTypes = {"csv": _loadCSV,
                      "xlsx": _readExcel,
                      "xlsm": _readExcel,
                      "xls": _readExcel}

    # Call function to open given file and return pandas dataframe
    if extension in validFileTypes:
        df = validFileTypes[extension](filePath, sep, headerRow)
    else:
        print("{} is not a valid File Type".format(extension))
        return None

    # Return specific column if provided as parameter
    if columnIndex:
        return df.iloc[:, columnIndex]
    else:
        return df