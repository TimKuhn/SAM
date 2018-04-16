from inputs.fileupload import loadFile
from recommenderModels.latentSemanticIndexing import lsiRecommendatition
from recommenderModels.nGramRecommender import nGramRecommendation

src = loadFile("../data/coa_target_src.xlsx", headerRow=1, columnIndex=0)

users = [user for user in src.iloc[:,0]]

items = list(set(list(set([str(item) for item in src.iloc[:,1]])) + list(set([str(item) for item in src.iloc[:,2]])) + list(set([str(item) for item in src.iloc[:,3]]))))

for user in users:
    print("="*50)
    print(user)
    lsi_result = lsiRecommendatition(user, items, top_n=3, n_components=50)
    nGram_result = nGramRecommendation(user, items, top_n=3)
    print("<-LSI->")
    for res in lsi_result:
        print(res)
    
    print("<-nGram->")
    for res in nGram_result:
        print(res)
    
