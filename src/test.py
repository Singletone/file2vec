import kit
import parameters
import extraction
import weeding
import processing


pathTo = kit.PathTo('IMDB', 'imdb')
wordFrequencyMap = parameters.loadWordRequencyMap(pathTo.wordFrequencyMap)


text = 'I can only assume its victim of the same issues as Fandango, Amazon and other sites where ratings are faked to boost numbers ahead of or because of ticket sales. Its a shame. I hope an honest and accurate ratings website appears to save us all no matter how hard that may be.'
text = extraction.clean(text)
print text
text = weeding.subsampleAndPrune(text, wordFrequencyMap, 0.3, 5)
print text
contextProvider = processing.WordContextProvider(text=text)
contexts = list(contextProvider.iterate(3))
print contexts