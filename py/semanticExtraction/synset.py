from nltk.corpus import wordnet as wn

synset1 = wn.synsets('liabilities')
synset2 = wn.synsets('assets')
synset3 = wn.synsets('cash')
synset4 = wn.synsets('reserve')


print(synset1)
print(synset2)
print(synset3)
print(synset4)