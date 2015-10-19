import math
import numpy


def cosineSimilarity(vectorA, vectorB):
    numerator = sum([a * b for a, b in zip(vectorA, vectorB)])
    denumerator = math.sqrt(sum([a * a for a in vectorA]) * sum([b * b for b in vectorB]))

    denumerator = 0.000000000001 if denumerator == 0 else denumerator

    return numerator / denumerator


def euclideanDistance(vectorA, vectorB):
    distance = numpy.power([vectorA - vectorB], 2).sum()
    distance = numpy.sqrt(distance)

    return distance