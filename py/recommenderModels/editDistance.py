"""


"""


from nltk.metrics import edit_distance


def _sortScoresAndReturnTopMatches(scores, top_n):
    """
    Returns the top_n results of scores

    Parameters:
    -----------
    scores: list of tuples, positional (e.g. [(0, 0.58), (1, 0.40), (2, 0.21)])
        List of results stored as a tuple where index 0 is the item and index 1 is the score
    top_n: int, positional
        Integer representing the number of scores that should be returned
    """
    return sorted(scores, key=lambda scores: scores[1])[:top_n]


def editStringDistance(user, items, top_n=5):
    """
    TODO: DOCSTRING

    Parameters:
    -----------
    user: string, positional (example: "a sequence of words")
        Look for recommendation to this user / word
    items: list, positional (example: ["item1", "item2", "item3"])
        Possible recommendatations in the form of a list of strings
    top_n: int, optional (default: 5)
        Number of recommendatitions returned from the model (e.g. 5 = top 5 results) 

    """

    # Calculate edit distance
    itemsAndeditDistances = [(item, edit_distance(user, item)) for item in items]

    # Calculate the relative adjustments needed compared to length of the user string
    itemsAndrelativeDistance = [(item, distance/len(user)) for item, distance in itemsAndeditDistances]

    # Sort and retrieve top n results
    topNitemIndexAndRelativeDistance = _sortScoresAndReturnTopMatches(itemsAndrelativeDistance, top_n)
    
    return topNitemIndexAndRelativeDistance

#result = editStringDistance("Hello", ["Hallo", "Hello", "hello", "Helloooo", "beer", "tim", "tom", "Helo"])

#print(result)