"""
nGramRecommender Model

Author: Tim Dilmaghani

"""


from nltk import ngrams

from sklearn import svm


#svm = svm.SVC()

#ngrams()


def _jaccardCoefficient(user, item):
    """ 
    Returns a floating point similarity score

    Calculates a matching score between 0 and 1 

    Defined as the size of the intersection 
    divided by the size of the union of the sample sets
    
    Parameters:
    -----------
    user: nGramList, positional
        #TODO
    item: nGramList, positional
        #TODO
    """

    user = set(user) 
    item = set(item)

    intersection = len(user.intersection(item))
    union = len(user.union(item))

    if union > 0:
        return intersection / union
    else:
        return 0

def _nGramCharTransformer(sequence, n=2, pad_left=False, pad_right=False): 
    """
    Returns a list of tuples where a tuple represents ngram

    Convert a string into its character ngram representation

    Parameters:
    -----------
    sequence: string, positional
        A string for conversion.
    n: int, optional (default: 2)
        Defines the numbers of ngrams (e.g. bigram = 2)
    pad_left: boolean, optional
        Left padding '<s>'
    pad_right: boolean, optional
        Right padding '</s>'
    """

    # Look-up table for padding symbols
    padding_symbols = {"pad_left": "<s>",
                       "pad_right": "</s>"}

    if pad_left:
        pad_left = padding_symbols[pad_left]

    if pad_right:
        pad_right = padding_symbols[pad_right]

    # Type checking and retrieving ngrams
    if isinstance(sequence, str):
        return list(ngrams(sequence, n, pad_left, pad_right))
    else:
        print("{} is of type [{}] but [str] needed".format(sequence, type(sequence)))
        return None

def _sortScoresAndReturnTopMatches(scores, top_n):
    """
    Returns the top_n results of scores

    Parameters:
    -----------
    scores: list of tuples, positional (e.g. [("item1", 0.58), ("item2", 0.40), ("item3", 0.21)])
        List of results stored as a tuple where index 0 is the item and index 1 is the score
    top_n: int, positional
        Integer representing the number of scores that should be returned
    """
    return sorted(scores, key=lambda scores: scores[1], reverse=True)[:top_n]

def _isCorrectType(user, items):
    """
    Type checking function

    """
    correctUserType = False
    correctItemsType = False
    correctItemsInputType = False

    if type(user) == str:
        correctUserType = True
    else:
        print("user is not of type [str]")

    if type(items) == list:
        correctItemsType = True

        subType = []
        for item in items:
            if type(item) == str:
                subType.append(True)
            else:
                print("{} is not of type [str]".format(type(item)))

        if False not in subType:
            correctItemsInputType = True

    if correctUserType and correctItemsType and correctItemsInputType:
        return True
    else:
        return False

def nGramRecommendation(user, items, top_n=5, n=2, pad_left=False, pad_right=False):
    """
    Returns a list of tuples [("item2", 0.44), ("item10", 0.32), ("item 7", 0.31)]

    Recommends top items to a specified user based on character nGrams

    Parameters:
    user: string, positional (example: "a sequence of words")
        Look for recommendation to this user / word
    items: list, positional (example: ["item1", "item2", "item3"])
        Possible recommendatations in the form of a list of strings
    top_n: int, optional (default: 5)
        Number of recommendatitions returned from the model (e.g. 5 = top 5 results)
    n: int, optional (default: 2)
        Defines the numbers of ngrams (e.g. for bigram = 2)
    pad_left: boolean, optional
        Left padding '<s>'
    pad_right: boolean, optional
        Right padding '</s>'
    """

    # Type checking
    if not _isCorrectType(user, items):
        print("Input type not correct")
        return None

    # Transform user to nGram Representation
    userNgram = _nGramCharTransformer(user, n, pad_left, pad_right)

    # Loop through items, transform to nGram, calculate jaccard score and store in list
    itemsAndSimScores = []
    for item in items:
        itemNgram = _nGramCharTransformer(item, n, pad_left, pad_right)
        similarityScore = _jaccardCoefficient(userNgram, itemNgram)
        itemsAndSimScores.append((item, similarityScore))

    # Sort and retrieve top n results
    topNitemsAndSimScores = _sortScoresAndReturnTopMatches(itemsAndSimScores, top_n)

    return topNitemsAndSimScores

testItems = ["Non-current assets", "Current assets beer", "Other assets", "more assets", "Assets for all", "beer for all", "all for nothing"]

result = nGramRecommendation("Current assets beer", testItems)

print(result)