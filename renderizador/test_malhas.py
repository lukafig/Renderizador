#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Confere a triangulação das malhas (projeto 1.3). Rode: python test_malhas.py"""

from gl import GL

assert GL.faixas([0, 1, 2, -1, 3, 4, 5, -1]) == [[0, 1, 2], [3, 4, 5]]
assert GL.faixas([0, 1, 2]) == [[0, 1, 2]]                 # sem -1 no final
assert GL.tira([0, 1, 2, 3]) == [0, 1, 2, 1, 2, 3]         # 4 vértices = 2 triângulos
assert GL.leque([0, 1, 2, 3, 4]) == [0, 1, 2, 0, 2, 3, 0, 3, 4]
assert GL.tira(range(3)) == [0, 1, 2] and GL.leque([0, 1, 2]) == [0, 1, 2]

print("ok")
