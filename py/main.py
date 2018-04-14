from fileupload import loadFile
from preprocessing import preProcessText

src = loadFile("../data/coa_target_src.xlsx", headerRow=1, columnIndex=0)

txt = """"Hello World this is an accounting related
 to topic so it will be awesome. 
 You know what I mean!"""

res = preProcessText(txt, stemmer=True)
