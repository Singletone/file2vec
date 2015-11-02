import struct
import numpy as np


def read(file, dtype, count):
    buffer = np.fromfile(file, dtype=dtype, count=count)

    if count == 1:
        return buffer[0]

    return buffer


def write(file, format, buffer):
    if isinstance(buffer, np.ndarray):
        buffer = buffer.flatten()
    elif not isinstance(buffer, list):
        buffer = [buffer]

    format = format.format(len(buffer))
    buffer = struct.pack(format, *buffer)

    file.write(buffer)


def readi(file, count=1):
    return read(file, 'int32', count)


def writei(file, buffer):
    write(file, '{0}i', buffer)


def readf(file, count=1):
    return read(file, 'float32', count)


def writef(file, buffer):
    write(file, '{0}f', buffer)


def reads(file, length):
    return file.read(length)


def writes(file, buffer):
    file.write(buffer)


def dumpMatrix(path, matrix):
    rows, columns = matrix.shape
    values = np.asarray(matrix).flatten()

    with open(path, 'wb+') as file:
        writei(file, rows)
        writei(file, columns)
        writef(file, values)


def loadMatrix(path):
    with open(path, 'rb') as file:
        rows = readi(file)
        columns = readi(file)
        count = rows * columns
        values = readf(file, count)
        matrix = np.reshape(values, (rows, columns))

        return matrix