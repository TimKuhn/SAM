from inputs.fileupload import loadFile
from preprocessing.preprocessing import preProcessText
from recommenderModels.latentSemanticIndexing import lsiRecommendatition
from recommenderModels.nGramRecommender import nGramRecommendation
from recommenderModels.editDistance import editStringDistance

from termcolor import cprint 

#TODO: If top1 result has distance of >0.9 does it indicate that a new 
#TODO: account should be added to the chart of accounts
#TODO: Bigram/Trigram combinations
#TODO: Add TAGS to accounts (e.g (assets|liab|equity), (IC|No IC), (short-term|long-term))


src = loadFile("../data/coa_target_src.xlsx", headerRow=1, columnIndex=0)

users = [user for user in src.iloc[:,0]]

items = list(set(list(set([str(item) for item in src.iloc[:,1]])) + list(set([str(item) for item in src.iloc[:,2]])) + list(set([str(item) for item in src.iloc[:,3]]))))

index2items = {i: item for i, item in enumerate(items)}

index2StemmedItems = {i: preProcessText(item, stemmer=True) for i, item in enumerate(items)}

stemmed2items = {stemmed: item for item, stemmed in zip(index2items.values(), index2StemmedItems.values())}

stemmedList = [stemmed for stemmed in stemmed2items.keys()]

print(list(index2StemmedItems.values()))
#print(stemmed2items)


"""
for user in users:
    print("="*50)
    print(user)
    userStemmed = preProcessText(user, stemmer=True)
    lsi_result = lsiRecommendatition(userStemmed, stemmedList, top_n=3, n_components=50)
    #nGram_result = nGramRecommendation(user, items, top_n=3)
    print("<-LSI->")
    for res in lsi_result:
        print(stemmed2items[res[0]], res[1])
"""


for user in users:
    print("="*50)
    cprint(user, color="blue")
    lsi_result = lsiRecommendatition(user, items, top_n=3, n_components=50)
    nGram_result = nGramRecommendation(user, items, top_n=3, n=4)
    edit_result = editStringDistance(user, items, top_n=3)

    cprint("<-LSI->", color="green")

    for res in lsi_result:
        print(res)
    
    cprint("<-nGram->", color="yellow")
    for res in nGram_result:
        print(res)

    cprint("<-EditDistance->", color="red")
    for res in edit_result:
        print(res)