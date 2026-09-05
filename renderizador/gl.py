#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Luka Figueiredo & Luiz Durand
Disciplina: Computação Gráfica
Data: 19/08/2026
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    matriz_camera = np.identity(4)       # mundo -> espaço da câmera
    matriz_perspectiva = np.identity(4)  # câmera -> espaço de recorte
    matriz_tela = np.identity(4)         # coordenadas normalizadas -> pixels
    pilha_modelo = [np.identity(4)]      # pilha de matrizes objeto -> mundo

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

        # Estado inicial: pilha zerada e uma câmera padrão, caso a cena não
        # traga um nó Viewpoint.
        GL.pilha_modelo = [np.identity(4)]
        GL.viewpoint([0.0, 0.0, 10.0], [0.0, 0.0, 1.0, 0.0], math.pi / 4)

    # -------------------------------------------------------------------------
    # Funções auxiliares usadas pelo rasterizador 2D
    # -------------------------------------------------------------------------

    @staticmethod
    def rgb8(colors, key="emissiveColor"):
        """Converte uma cor do X3D (floats de 0 a 1) para o formato RGB8 (0 a 255)."""
        color = colors[key] if colors and key in colors else [1.0, 1.0, 1.0]
        return [int(max(0.0, min(1.0, c)) * 255) for c in color[:3]]

    @staticmethod
    def draw_pixel(x, y, color):
        """Pinta um pixel, descartando o que cai fora da tela (clipping do framebuffer)."""
        if 0 <= x < GL.width and 0 <= y < GL.height:
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

    @staticmethod
    def line(x0, y0, x1, y1, color):
        """Rasteriza um segmento de reta pelo algoritmo DDA.

        Anda um pixel por vez no eixo dominante (o de maior variação) e interpola
        o outro eixo, garantindo que a linha fique contínua em qualquer inclinação.
        """
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy)))  # número de pixels do eixo dominante

        if steps == 0:  # segmento degenerado: cai dentro de um único pixel
            GL.draw_pixel(math.floor(x0), math.floor(y0), color)
            return

        x_inc = dx / steps
        y_inc = dy / steps
        for i in range(steps + 1):
            GL.draw_pixel(math.floor(x0 + i * x_inc), math.floor(y0 + i * y_inc), color)

    @staticmethod
    def edge(xa, ya, xb, yb, px, py):
        """Função de aresta: o sinal diz de que lado da reta (a->b) o ponto p está."""
        return (xb - xa) * (py - ya) - (yb - ya) * (px - xa)

    @staticmethod
    def triangle(x0, y0, x1, y1, x2, y2, color):
        """Rasteriza um triângulo preenchido pelo teste das funções de aresta.

        Percorre apenas a bounding box do triângulo (limitada à tela) e pinta o
        pixel cujo centro está do lado de dentro das três arestas.
        """
        # Garante orientação anti-horária, para que "dentro" seja sempre sinal >= 0
        if GL.edge(x0, y0, x1, y1, x2, y2) < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1

        # Bounding box recortada pelos limites da tela (evita varrer a tela inteira)
        xmin = max(0, math.floor(min(x0, x1, x2)))
        xmax = min(GL.width - 1, math.ceil(max(x0, x1, x2)))
        ymin = max(0, math.floor(min(y0, y1, y2)))
        ymax = min(GL.height - 1, math.ceil(max(y0, y1, y2)))

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                px, py = x + 0.5, y + 0.5  # amostra no centro do pixel
                if (GL.edge(x0, y0, x1, y1, px, py) >= 0 and
                        GL.edge(x1, y1, x2, y2, px, py) >= 0 and
                        GL.edge(x2, y2, x0, y0, px, py) >= 0):
                    GL.draw_pixel(x, y, color)

    # -------------------------------------------------------------------------
    # Nós do X3D
    # -------------------------------------------------------------------------

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # A lista point vem no formato [x0, y0, x1, y1, ...]; cada par vira um pixel.
        color = GL.rgb8(colors)
        for i in range(0, len(point) - 1, 2):
            GL.draw_pixel(math.floor(point[i]), math.floor(point[i + 1]), color)

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # A lista vem como [x0, y0, x1, y1, ...] e forma uma polilinha: cada par de
        # pontos consecutivos é um segmento (n pontos geram n-1 segmentos).
        color = GL.rgb8(colors)
        for i in range(0, len(lineSegments) - 3, 2):
            GL.line(lineSegments[i], lineSegments[i + 1],
                    lineSegments[i + 2], lineSegments[i + 3], color)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Desenha o contorno de um círculo centrado na origem pelo algoritmo do
        # ponto médio (Bresenham): calcula 1/8 do círculo e espelha nos 8 octantes.
        color = GL.rgb8(colors)
        r = int(round(radius))

        x, y = 0, r
        d = 1 - r  # variável de decisão do ponto médio
        while x <= y:
            for px, py in ((x, y), (y, x), (y, -x), (x, -y),
                           (-x, -y), (-y, -x), (-y, x), (-x, y)):
                GL.draw_pixel(px, py, color)
            if d < 0:            # ponto médio dentro do círculo: mantém y
                d += 2 * x + 3
            else:                # ponto médio fora do círculo: desce um y
                d += 2 * (x - y) + 5
                y -= 1
            x += 1

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # A lista vem como [x0, y0, x1, y1, x2, y2, ...]; cada 6 valores (3 pontos)
        # formam um triângulo independente.
        color = GL.rgb8(colors)
        for i in range(0, len(vertices) - 5, 6):
            GL.triangle(vertices[i], vertices[i + 1],
                        vertices[i + 2], vertices[i + 3],
                        vertices[i + 4], vertices[i + 5], color)


    # -------------------------------------------------------------------------
    # Matrizes do pipeline 3D
    # -------------------------------------------------------------------------

    @staticmethod
    def matriz_translacao(t):
        """Matriz 4x4 de translação por t = [x, y, z]."""
        return np.array([[1.0, 0.0, 0.0, t[0]],
                         [0.0, 1.0, 0.0, t[1]],
                         [0.0, 0.0, 1.0, t[2]],
                         [0.0, 0.0, 0.0, 1.0]])

    @staticmethod
    def matriz_escala(s):
        """Matriz 4x4 de escala por s = [x, y, z]."""
        return np.array([[s[0], 0.0, 0.0, 0.0],
                         [0.0, s[1], 0.0, 0.0],
                         [0.0, 0.0, s[2], 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    @staticmethod
    def matriz_rotacao(r):
        """Matriz 4x4 de rotação a partir de r = [x, y, z, t] (eixo + ângulo em radianos).

        Usa a fórmula de Rodrigues, que resolve um eixo qualquer de uma vez só,
        em vez de compor rotações separadas em x, y e z.
        """
        eixo = np.array(r[:3], dtype=float)
        norma = np.linalg.norm(eixo)
        if norma == 0:  # eixo degenerado: rotação não definida, devolve identidade
            return np.identity(4)
        x, y, z = eixo / norma

        c = math.cos(r[3])
        s = math.sin(r[3])
        t = 1.0 - c
        return np.array([[t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0.0],
                         [t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0.0],
                         [t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0.0],
                         [0.0,         0.0,         0.0,         1.0]])

    @staticmethod
    def projetar(pontos):
        """Leva pontos do espaço do objeto para coordenadas de tela (em pixels).

        Aplica, nesta ordem: matriz do modelo (topo da pilha) -> câmera ->
        projeção perspectiva -> divisão perspectiva -> matriz de tela.
        Recebe e devolve uma matriz 4xN (cada coluna é um ponto homogêneo).
        """
        mvp = np.matmul(GL.matriz_perspectiva,
                        np.matmul(GL.matriz_camera, GL.pilha_modelo[-1]))
        p = np.matmul(mvp, pontos)

        # Divisão perspectiva: é ela que faz o objeto distante parecer menor
        w = np.where(np.abs(p[3]) < 1e-12, 1e-12, p[3])  # evita divisão por zero
        p = p / w

        return np.matmul(GL.matriz_tela, p)

    # -------------------------------------------------------------------------
    # Nós do X3D
    # -------------------------------------------------------------------------

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # A lista point vem como [x0, y0, z0, x1, y1, z1, ...]; cada 9 valores
        # (3 pontos) formam um triângulo independente.
        color = GL.rgb8(colors)

        # Monta todos os vértices de uma vez como matriz homogênea 4xN
        vertices = np.array(point, dtype=float).reshape(-1, 3).T
        vertices = np.vstack([vertices, np.ones(vertices.shape[1])])

        tela = GL.projetar(vertices)

        # Depois da projeção sobra um problema 2D, resolvido pelo mesmo
        # rasterizador de triângulos do projeto anterior
        for i in range(0, tela.shape[1] - 2, 3):
            GL.triangle(tela[0][i],     tela[1][i],
                        tela[0][i + 1], tela[1][i + 1],
                        tela[0][i + 2], tela[1][i + 2], color)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # A câmera é um objeto como outro qualquer: para levar o mundo para o
        # espaço da câmera aplica-se a transformação INVERSA da câmera. Como a
        # rotação é ortonormal, sua inversa é a transposta, e a translação
        # inversa é o negativo da posição.
        rotacao = GL.matriz_rotacao(orientation)
        GL.matriz_camera = np.matmul(
            rotacao.T, GL.matriz_translacao(-np.array(position, dtype=float)))

        # Projeção perspectiva. No X3D fieldOfView se refere à menor dimensão da
        # tela, então o ângulo é reescalado pela diagonal do viewport.
        fovy = 2 * math.atan(math.tan(fieldOfView / 2) * GL.height /
                             math.hypot(GL.width, GL.height))
        top = GL.near * math.tan(fovy)
        right = top * (GL.width / GL.height)

        GL.matriz_perspectiva = np.array([
            [GL.near / right, 0.0, 0.0, 0.0],
            [0.0, GL.near / top, 0.0, 0.0],
            [0.0, 0.0, -(GL.far + GL.near) / (GL.far - GL.near),
             -2.0 * GL.far * GL.near / (GL.far - GL.near)],
            [0.0, 0.0, -1.0, 0.0]])

        # Matriz de tela: leva as coordenadas normalizadas (-1 a 1) para pixels.
        # O -height/2 inverte o eixo y, que no X3D cresce para cima e na tela
        # cresce para baixo.
        GL.matriz_tela = np.array([[GL.width / 2, 0.0, 0.0, GL.width / 2],
                                   [0.0, -GL.height / 2, 0.0, GL.height / 2],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # Monta a matriz do modelo na ordem escala -> rotação -> translação
        # (lida da direita para a esquerda no produto de matrizes).
        matriz = np.identity(4)
        if translation:
            matriz = np.matmul(matriz, GL.matriz_translacao(translation))
        if rotation:
            matriz = np.matmul(matriz, GL.matriz_rotacao(rotation))
        if scale:
            matriz = np.matmul(matriz, GL.matriz_escala(scale))

        # Empilha já combinada com a matriz do nó pai, para suportar Transforms
        # aninhados: o topo da pilha é sempre "objeto -> mundo".
        GL.pilha_modelo.append(np.matmul(GL.pilha_modelo[-1], matriz))

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # Ao sair do nó, desempilha para voltar ao referencial do pai.
        if len(GL.pilha_modelo) > 1:
            GL.pilha_modelo.pop()


    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TriangleStripSet : pontos = {0} ".format(point), end='')
        for i, strip in enumerate(stripCount):
            print("strip[{0}] = {1} ".format(i, strip), end='')
        print("")
        print("TriangleStripSet : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedFaceSet : ")
        if coord:
            print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        print("colorPerVertex = {0}".format(colorPerVertex))
        if colorPerVertex and color and colorIndex:
            print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        if texCoord and texCoordIndex:
            print("\tpontos(u, v) = {0}, texCoordIndex = {1}".format(texCoord, texCoordIndex))
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            print("\t Matriz com image = {0}".format(image))
            print("\t Dimensões da image = {0}".format(image.shape))
        print("IndexedFaceSet : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
