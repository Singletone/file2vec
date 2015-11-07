import kit
import parameters
import extraction
import weeding

pathTo = kit.PathTo('IMDB', 'imdb')
wordFrequencyMap = parameters.loadWordRequencyMap(pathTo.wordFrequencyMap)

text = 'I love that they have kept the original style of the site and only changed little details over the years.'
text = extraction.clean(text)

sample = 0.3

print text
print weeding.subsampleAndPrune(text, wordFrequencyMap, sample, 5)