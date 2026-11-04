"""Small dependency-free Cauchy Reed-Solomon codec for packet erasures."""

import numpy as np


def _tables():
    exp = np.zeros(512, dtype=np.uint8)
    log = np.zeros(256, dtype=np.int16)
    value = 1
    for index in range(255):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x100:
            value ^= 0x11D
    exp[255:] = exp[:257]
    mul = np.zeros((256, 256), dtype=np.uint8)
    for left in range(1, 256):
        for right in range(1, 256):
            mul[left, right] = exp[int(log[left]) + int(log[right])]
    return exp, log, mul


GF_EXP, GF_LOG, GF_MUL = _tables()


def gf_inv(value):
    if not value:
        raise ZeroDivisionError("zero has no multiplicative inverse")
    return int(GF_EXP[255 - int(GF_LOG[value])])


def cauchy_coefficients(data_count, parity_count):
    if data_count < 1 or parity_count < 1 or data_count + parity_count > 255:
        raise ValueError("Cauchy RS requires 1 <= data+parity <= 255")
    matrix = np.empty((parity_count, data_count), dtype=np.uint8)
    for row in range(parity_count):
        for column in range(data_count):
            matrix[row, column] = gf_inv(row ^ (parity_count + column))
    return matrix


def encode_group(data_packets, parity_count):
    if not data_packets:
        raise ValueError("Cannot encode an empty packet group")
    width = len(data_packets[0])
    if any(len(packet) != width for packet in data_packets):
        raise ValueError("All data packets must have equal length")
    data = [np.asarray(packet, dtype=np.uint8) for packet in data_packets]
    coefficients = cauchy_coefficients(len(data), parity_count)
    parity = []
    for row in range(parity_count):
        result = np.zeros(width, dtype=np.uint8)
        for column, packet in enumerate(data):
            result ^= GF_MUL[int(coefficients[row, column]), packet]
        parity.append(result)
    return parity


def _solve(matrix, rhs):
    matrix = np.asarray(matrix, dtype=np.uint8).copy()
    rhs = [np.asarray(value, dtype=np.uint8).copy() for value in rhs]
    size = len(rhs)
    for column in range(size):
        pivot = next((row for row in range(column, size)
                      if matrix[row, column]), None)
        if pivot is None:
            raise ValueError("Singular Reed-Solomon recovery matrix")
        if pivot != column:
            matrix[[column, pivot]] = matrix[[pivot, column]]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        inverse = gf_inv(int(matrix[column, column]))
        matrix[column] = GF_MUL[inverse, matrix[column]]
        rhs[column] = GF_MUL[inverse, rhs[column]]
        for row in range(size):
            if row == column:
                continue
            factor = int(matrix[row, column])
            if factor:
                matrix[row] ^= GF_MUL[factor, matrix[column]]
                rhs[row] ^= GF_MUL[factor, rhs[column]]
    return rhs


def recover_group(data_packets, parity_packets):
    """Recover missing data packets represented by ``None`` values."""
    missing = [index for index, packet in enumerate(data_packets)
               if packet is None]
    if not missing:
        return list(data_packets)
    available_rows = [index for index, packet in enumerate(parity_packets)
                      if packet is not None]
    if len(available_rows) < len(missing):
        raise ValueError("Not enough surviving parity packets")
    surviving = next((packet for packet in data_packets if packet is not None),
                     next((packet for packet in parity_packets
                           if packet is not None), None))
    if surviving is None:
        raise ValueError("No packet data remains")
    coefficients = cauchy_coefficients(len(data_packets), len(parity_packets))
    selected = available_rows[:len(missing)]
    rhs = []
    for row in selected:
        value = np.asarray(parity_packets[row], dtype=np.uint8).copy()
        for column, packet in enumerate(data_packets):
            if packet is not None:
                value ^= GF_MUL[int(coefficients[row, column]),
                                np.asarray(packet, dtype=np.uint8)]
        rhs.append(value)
    matrix = coefficients[np.ix_(selected, missing)]
    solved = _solve(matrix, rhs)
    recovered = list(data_packets)
    for column, packet in zip(missing, solved):
        recovered[column] = packet
    return recovered
