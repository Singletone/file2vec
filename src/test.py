import processing

text = 'robotboy found episode show teach behave. main.'

contextProvider = processing.WordContextProvider(text, count=600)

for context in contextProvider.iterate(3):
    print context