"""Tudo que é desenho na tela (OpenCV), no tema "pink tech glam": gradiente rosa
com grade, painéis de vidro, glitter animado, peças brilhantes e esqueleto neon.

Lê o estado do `Jogo` mas nunca o altera. As fontes do OpenCV não têm acentos,
então todo texto passa por `_ascii` (ex.: "câmera" -> "camera").
Todos os desenhos são primitivas do OpenCV (nenhuma imagem de terceiros).
"""
from __future__ import annotations

import math
import random
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from logica import (Estado, Evento, Jogo, MotivoPausa, TipoEvento, TipoItem,
                    TipoObstaculo)
from modelos import Calibracao, Corpo
from pose import Leitura

FONTE = cv2.FONT_HERSHEY_DUPLEX
FONTE_SCRIPT = cv2.FONT_HERSHEY_SCRIPT_COMPLEX   # títulos "cursivos" glam
Cor = Tuple[int, int, int]

# Ligações do esqueleto (topologia do BlazePose; ver documentação do MediaPipe):
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
CONEXOES = [
    (0, 2), (0, 5), (2, 7), (5, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 19), (16, 20), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32),
]


_DESLOC_CONTORNO = [(-2, -1), (2, -1), (-2, 2), (2, 2)]
_DESLOC_HALO = [(-4, 0), (4, 0), (0, -4), (0, 4)]


def _agora() -> float:
    return time.perf_counter()


# ==========================================================================
# Primitivas
# ==========================================================================
def _ascii(txt: str) -> str:
    return unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")


def texto(img: np.ndarray, txt: str, x: int, y: int, escala: float = 0.6,
          cor: Cor = cfg.COR_BRANCO, esp: int = 1, centro: bool = False,
          direita: bool = False, fonte: int = FONTE, brilho: bool = False) -> None:
    """Texto com contorno ameixa; `brilho=True` acrescenta um halo rosa choque."""
    txt = _ascii(txt)
    (w, _), _ = cv2.getTextSize(txt, fonte, escala, esp)
    if centro:
        x -= w // 2
    elif direita:
        x -= w
    # Contorno/halo feito com cópias deslocadas do texto (espessura engrossada
    # no putText desalinha o glifo e deixa uma "sombra" para o lado).
    if brilho:
        for dx, dy in _DESLOC_HALO:
            cv2.putText(img, txt, (x + dx, y + dy), fonte, escala, cfg.COR_ROSA, esp,
                        cv2.LINE_AA)
    for dx, dy in _DESLOC_CONTORNO:
        cv2.putText(img, txt, (x + dx, y + dy), fonte, escala, cfg.COR_AMEIXA, esp, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), fonte, escala, cor, esp, cv2.LINE_AA)


def _estrela(img: np.ndarray, x: float, y: float, r: float, cor: Cor,
             nucleo: Cor = cfg.COR_BRANCO) -> None:
    """Brilho de 4 pontas (o "glitter")."""
    if r < 1:
        return
    f = 0.22 * r
    pts = np.array([[x, y - r], [x + f, y - f], [x + r, y], [x + f, y + f],
                    [x, y + r], [x - f, y + f], [x - r, y], [x - f, y - f]], np.int32)
    cv2.fillPoly(img, [pts], cor, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), max(1, int(r * 0.18)), nucleo, -1, cv2.LINE_AA)


def _retangulo_arredondado(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                           cor: Cor, esp: int = 2, r: int = 14) -> None:
    """Retângulo de cantos arredondados (esp < 0 = preenchido)."""
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if esp < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), cor, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), cor, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, cor, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), cor, esp, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), cor, esp, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), cor, esp, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), cor, esp, cv2.LINE_AA)
    for (cx, cy), a0 in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                         ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, cor, esp, cv2.LINE_AA)


def _vidro(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, alfa: float = 0.55,
           r: int = 14, borda: bool = True) -> None:
    """Painel de "vidro" rosa translúcido com borda neon."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(2, x1), max(2, y1), min(w - 3, x2), min(h - 3, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = img[y1:y2, x1:x2]
    mascara = np.zeros(roi.shape[:2], np.uint8)
    _retangulo_arredondado(mascara, 0, 0, x2 - x1 - 1, y2 - y1 - 1, 255, -1, r)
    cor = np.full_like(roi, (120, 30, 150))
    mistura = cv2.addWeighted(roi, 1 - alfa, cor, alfa, 0)
    roi[mascara > 0] = mistura[mascara > 0]
    if borda:
        _retangulo_arredondado(img, x1, y1, x2, y2, cfg.COR_ROSA, 4, r)
        _retangulo_arredondado(img, x1, y1, x2, y2, cfg.COR_BRANCO, 1, r)


def _tracejada(img: np.ndarray, y: int, cor: Cor, espessura: int = 1) -> None:
    for x in range(0, img.shape[1], 24):
        cv2.line(img, (x, y), (x + 12, y), cor, espessura, cv2.LINE_AA)


def _coracao(img: np.ndarray, cx: int, cy: int, r: int, cor: Cor, cheio: bool = True) -> None:
    """Coração com brilho. cheio=False desenha só o contorno (vida perdida)."""
    esp = -1 if cheio else 2
    cv2.circle(img, (cx - r // 2, cy - r // 3), r // 2 + 1, cor, esp, cv2.LINE_AA)
    cv2.circle(img, (cx + r // 2, cy - r // 3), r // 2 + 1, cor, esp, cv2.LINE_AA)
    pts = np.array([[cx - r, cy - r // 6], [cx + r, cy - r // 6], [cx, cy + r]], np.int32)
    if cheio:
        cv2.fillConvexPoly(img, pts, cor, cv2.LINE_AA)
        cv2.circle(img, (cx - r // 2, cy - r // 2), max(1, r // 5), cfg.COR_BRANCO, -1, cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], True, cor, 2, cv2.LINE_AA)


def _diamante(img: np.ndarray, cx: int, cy: int, r: int, cor: Cor = cfg.COR_CIANO) -> None:
    """Gema facetada (ícone do escudo)."""
    pts = np.array([[cx - r, cy - r // 3], [cx - r // 2, cy - r], [cx + r // 2, cy - r],
                    [cx + r, cy - r // 3], [cx, cy + r]], np.int32)
    cv2.fillPoly(img, [pts], cor, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, cfg.COR_BRANCO, 1, cv2.LINE_AA)
    cv2.line(img, (cx - r, cy - r // 3), (cx + r, cy - r // 3), cfg.COR_BRANCO, 1, cv2.LINE_AA)
    cv2.line(img, (cx - r // 2, cy - r), (cx, cy + r), cfg.COR_BRANCO, 1, cv2.LINE_AA)
    cv2.line(img, (cx + r // 2, cy - r), (cx, cy + r), cfg.COR_BRANCO, 1, cv2.LINE_AA)


def _glitter(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, n: int, semente: int) -> None:
    """Pontinhos brilhantes que piscam dentro de uma área (peças com glitter)."""
    rng = random.Random(semente)
    t = _agora()
    for i in range(n):
        px, py = rng.uniform(x1, x2), rng.uniform(y1, y2)
        brilho = 0.5 + 0.5 * math.sin(t * 6 + i * 1.7)
        if brilho > 0.55 and 0 <= px < img.shape[1]:
            _estrela(img, px, py, 2 + 5 * brilho, cfg.COR_BRANCO, cfg.COR_ROSA_CLARO)


# ==========================================================================
# Fundo "pink tech": gradiente + grade + vinheta (pré-calculado uma vez)
# ==========================================================================
_CACHE_FUNDO: Dict[Tuple[int, int], np.ndarray] = {}


def _criar_fundo(h: int, w: int) -> np.ndarray:
    ys = np.linspace(0.0, 1.0, h)[:, None, None]
    topo = np.array((200, 90, 255), np.float32)    # rosa choque (BGR)
    base = np.array((210, 40, 130), np.float32)    # roxo profundo
    fundo = (topo * (1 - ys) + base * ys) * np.ones((1, w, 1), np.float32)
    fundo = fundo.astype(np.uint8)
    for x in range(0, w, 40):                       # grade vertical
        cv2.line(fundo, (x, 0), (x, h), (255, 215, 255), 1)
    for y in range(0, h, 40):                       # grade horizontal
        cv2.line(fundo, (0, y), (w, y), (255, 215, 255), 1)
    # vinheta: bordas mais escuras
    xs = np.linspace(-1, 1, w)[None, :]
    yv = np.linspace(-1, 1, h)[:, None]
    vinheta = np.clip(1.15 - 0.45 * (xs ** 2 + yv ** 2), 0.5, 1.0)[:, :, None]
    return (fundo * vinheta).astype(np.uint8)


def aplicar_fundo(img: np.ndarray) -> None:
    chave = img.shape[:2]
    if chave not in _CACHE_FUNDO:
        _CACHE_FUNDO[chave] = _criar_fundo(*chave)
    img[:] = cv2.addWeighted(img, 0.68, _CACHE_FUNDO[chave], 0.32, 0)


# ==========================================================================
# Efeitos visuais (glitter ambiente, dano, coleta, textos flutuantes)
# ==========================================================================
class Efeitos:
    """Estado puramente visual: reage aos eventos da lógica."""

    def __init__(self) -> None:
        self.rng = random.Random()
        self._flash = 0.0
        self._flash_dur = 0.45
        self._flash_cor: Cor = cfg.COR_VERMELHO
        self._textos: List[list] = []        # [txt, x, y, vida, cor, escala]
        self._particulas: List[list] = []    # [x, y, vx, vy, vida, cor, forma]
        # glitter ambiente: [x, y, vy, fase, tamanho]
        self._ambiente = [[self.rng.uniform(0, cfg.LARGURA), self.rng.uniform(0, cfg.ALTURA),
                           self.rng.uniform(8, 30), self.rng.uniform(0, 6.28),
                           self.rng.uniform(2, 6)] for _ in range(26)]

    def processar(self, eventos: List[Evento]) -> None:
        for e in eventos:
            if e.tipo == TipoEvento.DANO:
                self._flash, self._flash_cor, self._flash_dur = 0.5, cfg.COR_VERMELHO, 0.5
                self._texto("OPS! -1 VIDA", e.x, e.y, cfg.COR_BRANCO, 0.85)
                self._explodir(e.x, e.y, cfg.COR_ROSA, 26, "coracao")
            elif e.tipo == TipoEvento.ESCUDO_USADO:
                self._flash, self._flash_cor, self._flash_dur = 0.35, cfg.COR_CIANO, 0.35
                self._texto("PROTEGIDA!", e.x, e.y, cfg.COR_CIANO, 0.85)
                self._explodir(e.x, e.y, cfg.COR_CIANO, 22, "estrela")
            elif e.tipo == TipoEvento.MOEDA:
                self._texto(f"+{e.valor}", e.x, e.y, cfg.COR_DOURADO, 0.8)
                self._explodir(e.x, e.y, cfg.COR_DOURADO, 20, "estrela")
            elif e.tipo == TipoEvento.ESCUDO:
                self._texto(f"GEMA +{e.valor}", e.x, e.y, cfg.COR_CIANO, 0.7)
                self._explodir(e.x, e.y, cfg.COR_CIANO, 20, "estrela")
            elif e.tipo == TipoEvento.ESQUIVA:
                self._texto(f"+{e.valor}", e.x, e.y, cfg.COR_VERDE, 0.7)
            elif e.tipo == TipoEvento.RASPAO:
                self._texto(f"RASPAO! +{e.valor}", e.x, e.y, cfg.COR_LARANJA, 0.8)
                self._explodir(e.x, e.y + 20, cfg.COR_BRANCO, 14, "estrela")
            elif e.tipo == TipoEvento.COMBO_PERDIDO:
                self._texto("combo perdido", e.x, e.y - 30, cfg.COR_CINZA, 0.55)
            elif e.tipo == TipoEvento.FASE:
                for _ in range(3):
                    self._explodir(self.rng.uniform(80, cfg.LARGURA - 80),
                                   self.rng.uniform(120, 260), self.rng.choice(
                                       (cfg.COR_ROSA, cfg.COR_CIANO, cfg.COR_DOURADO)), 14, "estrela")

    def _texto(self, txt: str, x: float, y: float, cor: Cor, escala: float) -> None:
        self._textos.append([txt, x, y, 1.1, cor, escala])

    def _explodir(self, x: float, y: float, cor: Cor, n: int, forma: str) -> None:
        for _ in range(n):
            ang = self.rng.uniform(0, 2 * math.pi)
            vel = self.rng.uniform(60, 260)
            self._particulas.append([x, y, math.cos(ang) * vel, math.sin(ang) * vel,
                                     self.rng.uniform(0.4, 0.9), cor, forma])

    def atualizar(self, dt: float) -> None:
        self._flash = max(0.0, self._flash - dt)
        for a in self._ambiente:
            a[1] += a[2] * dt
            if a[1] > cfg.ALTURA + 8:
                a[0], a[1] = self.rng.uniform(0, cfg.LARGURA), -8.0
        for t in self._textos:
            t[2] -= 45 * dt
            t[3] -= dt
        self._textos = [t for t in self._textos if t[3] > 0]
        for p in self._particulas:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 300 * dt
            p[4] -= dt
        self._particulas = [p for p in self._particulas if p[4] > 0]

    def desenhar_ambiente(self, img: np.ndarray) -> None:
        """Glitter caindo devagar por trás de tudo."""
        t = _agora()
        for x, y, _, fase, tam in self._ambiente:
            brilho = 0.5 + 0.5 * math.sin(t * 3 + fase)
            _estrela(img, x, y, tam * (0.5 + brilho), cfg.COR_ROSA_CLARO, cfg.COR_BRANCO)

    def desenhar(self, img: np.ndarray) -> None:
        for x, y, _, _, vida, cor, forma in self._particulas:
            if forma == "coracao":
                _coracao(img, int(x), int(y), max(3, int(vida * 9)), cor)
            else:
                _estrela(img, x, y, max(2.0, vida * 12), cor)
        for txt, x, y, _, cor, escala in self._textos:
            texto(img, txt, int(x), int(y), escala, cor, 2, centro=True, brilho=True)
        if self._flash > 0:   # flash de tela cheia, some gradualmente
            a = 0.45 * self._flash / self._flash_dur
            camada = np.full_like(img, self._flash_cor)
            img[:] = cv2.addWeighted(img, 1 - a, camada, a, 0)
            esp = int(4 + 14 * self._flash / self._flash_dur)
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), self._flash_cor, esp)


# ==========================================================================
# Elementos da cena
# ==========================================================================
def desenhar_zonas(img: np.ndarray, cal: Calibracao, debug: bool) -> None:
    """Linhas das zonas de AGACHAR e PULAR, definidas pela calibração."""
    L = cal.tronco
    y_agachar = int(cal.ombro_y + cfg.AGACHAR_ENTRA_FRAC * L)
    y_pular = int(cal.quadril_y - cfg.PULAR_ENTRA_FRAC * L)
    esp = 2 if debug else 1
    _tracejada(img, y_agachar, cfg.COR_DOURADO, esp)
    _tracejada(img, y_pular, cfg.COR_CIANO, esp)
    texto(img, "ombros abaixo = AGACHAR", 6, y_agachar + 16, 0.4, cfg.COR_DOURADO)
    texto(img, "quadril acima = PULAR", 6, y_pular - 6, 0.4, cfg.COR_CIANO)


def desenhar_obstaculo(img: np.ndarray, o, tempo: float) -> None:
    x, y, w, h = int(o.x), int(o.y), int(o.w), int(o.h)
    if o.tipo == TipoObstaculo.ALTO:
        # barra laser pink: halo, corpo rosa, faixa de brilho e pontas em ciano
        cv2.rectangle(img, (x - 4, y), (x + w + 4, y + h + 4), cfg.COR_ROSA, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), cfg.COR_PINK_FORTE, -1)
        cv2.rectangle(img, (x + 8, y), (x + 20, y + h), cfg.COR_ROSA, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), cfg.COR_BRANCO, 2)
        for i in range(0, w, 14):
            cv2.fillConvexPoly(img, np.array([[x + i, y + h], [x + i + 14, y + h],
                                              [x + i + 7, y + h + 14]], np.int32),
                               cfg.COR_CIANO, cv2.LINE_AA)
        _glitter(img, x, y, x + w, y + h, 8, id(o) % 1000)
        texto(img, "ABAIXE", x + w // 2, max(20, y + h - 12), 0.45, cfg.COR_BRANCO, centro=True)
    elif o.tipo == TipoObstaculo.BAIXO:
        # caixa de presente glam: corpo pink, laço de brilho e pontas em ciano
        cv2.rectangle(img, (x - 4, y - 4), (x + w + 4, y + h), cfg.COR_ROSA, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), cfg.COR_PINK_FORTE, -1)
        cv2.rectangle(img, (x + w // 2 - 6, y), (x + w // 2 + 6, y + h), cfg.COR_ROSA_CLARO, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), cfg.COR_BRANCO, 2)
        for i in range(0, w, 14):
            cv2.fillConvexPoly(img, np.array([[x + i, y], [x + i + 14, y],
                                              [x + i + 7, y - 14]], np.int32),
                               cfg.COR_CIANO, cv2.LINE_AA)
        _glitter(img, x, y, x + w, min(y + h, img.shape[0]), 8, id(o) % 1000)
        texto(img, "PULE", x + w // 2, y + 30, 0.5, cfg.COR_BRANCO, 1, centro=True)
    else:  # LATERAL
        if o.aviso > 0:   # coluna de alerta piscando onde o cristal vai cair
            if int(tempo * 8) % 2 == 0:
                _vidro(img, x, 0, x + w, img.shape[0], 0.35, 4)
            seta = "INCLINE ->" if o.lado < 0 else "<- INCLINE"
            texto(img, seta, x + w // 2, 60, 0.55, cfg.COR_DOURADO, 2, centro=True, brilho=True)
            _estrela(img, x + w // 2, 28, 12 + 4 * math.sin(tempo * 10), cfg.COR_BRANCO,
                     cfg.COR_ROSA)
        else:   # cristal rosa facetado
            pts = np.array([[x + w // 2, y], [x + w, y + int(h * 0.3)], [x + int(w * 0.85), y + h],
                            [x + int(w * 0.15), y + h], [x, y + int(h * 0.3)]], np.int32)
            cv2.fillPoly(img, [pts], cfg.COR_ROSA, cv2.LINE_AA)
            cv2.fillPoly(img, [np.array([pts[0], pts[1], [x + w // 2, y + h // 2]], np.int32)],
                         cfg.COR_ROSA_CLARO, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, cfg.COR_BRANCO, 2, cv2.LINE_AA)
            cv2.line(img, (x + w // 2, y), (x + w // 2, y + h), cfg.COR_BRANCO, 1, cv2.LINE_AA)
            cv2.line(img, (x, y + int(h * 0.3)), (x + w, y + int(h * 0.3)), cfg.COR_BRANCO, 1,
                     cv2.LINE_AA)
            _glitter(img, x, y, x + w, y + h, 7, id(o) % 1000)


def desenhar_item(img: np.ndarray, it, tempo: float) -> None:
    x, y, r = int(it.x), int(it.y), int(it.raio)
    pulso = int(2 * math.sin(tempo * 8))
    if it.tipo == TipoItem.MOEDA:   # moeda dourada com estrela
        cv2.circle(img, (x, y), r + 5 + pulso, cfg.COR_ROSA, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), r + pulso, cfg.COR_DOURADO, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r + pulso, cfg.COR_BRANCO, 2, cv2.LINE_AA)
        _estrela(img, x, y, r * 0.65, cfg.COR_BRANCO, cfg.COR_DOURADO)
    else:                            # gema de proteção
        cv2.circle(img, (x, y), r + 6 + pulso, cfg.COR_CIANO, 2, cv2.LINE_AA)
        _diamante(img, x, y, r - 2)


def desenhar_esqueleto(img: np.ndarray, leitura: Leitura) -> None:
    """Esqueleto neon: halo rosa por baixo e linha clara por cima."""
    if leitura.pontos is None:
        return
    ok = leitura.presente
    halo = cfg.COR_PINK_FORTE if ok else cfg.COR_VERMELHO
    linha = cfg.COR_ESQUELETO if ok else cfg.COR_ROSA_CLARO
    p = leitura.pontos
    ligacoes = [(a, b) for a, b in CONEXOES
                if p[a][2] >= cfg.LIMIAR_VIS_OPCIONAL and p[b][2] >= cfg.LIMIAR_VIS_OPCIONAL]
    for a, b in ligacoes:
        cv2.line(img, (int(p[a][0]), int(p[a][1])), (int(p[b][0]), int(p[b][1])),
                 halo, 9, cv2.LINE_AA)
    for a, b in ligacoes:
        cv2.line(img, (int(p[a][0]), int(p[a][1])), (int(p[b][0]), int(p[b][1])),
                 linha, 3, cv2.LINE_AA)
    for i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        if p[i][2] >= cfg.LIMIAR_VIS_OPCIONAL:
            c = (int(p[i][0]), int(p[i][1]))
            cv2.circle(img, c, 7, cfg.COR_ROSA, -1, cv2.LINE_AA)
            cv2.circle(img, c, 4, cfg.COR_BRANCO, -1, cv2.LINE_AA)


def desenhar_caixas(img: np.ndarray, corpo: Corpo, invulneravel: float, tempo: float) -> None:
    """Zonas de colisão do corpo: cabeça e tronco (piscam em branco ao tomar dano)."""
    piscando = invulneravel > 0 and int(tempo * 10) % 2 == 0
    for caixa, cor, nome in ((corpo.cabeca, cfg.COR_CABECA, "CABECA"),
                             (corpo.tronco, cfg.COR_TRONCO, "TRONCO")):
        c = cfg.COR_BRANCO if piscando else cor
        p1, p2 = (int(caixa.x1), int(caixa.y1)), (int(caixa.x2), int(caixa.y2))
        _retangulo_arredondado(img, p1[0], p1[1], p2[0], p2[1], c, 2, 10)
        texto(img, nome, p1[0] + 4, p1[1] - 5, 0.35, c)
    for mx, my in corpo.maos:   # mãos: pontos de coleta de itens
        cv2.circle(img, (int(mx), int(my)), cfg.MAO_RAIO_PX, cfg.COR_DOURADO, 2, cv2.LINE_AA)
        _estrela(img, mx, my, 6, cfg.COR_DOURADO)


def desenhar_acoes(img: np.ndarray, corpo: Corpo) -> None:
    """Mostra o que a aplicação ENTENDEU: ação(ões) reconhecida(s)."""
    a = corpo.acoes
    nomes: List[str] = []
    if a.agachado:
        nomes.append("AGACHADO")
    if a.pulando:
        nomes.append("PULANDO")
    if a.inclinacao:
        nomes.append("INCLINADO")
    if a.mao_esq_alta or a.mao_dir_alta:
        nomes.append("MAO ALTA")
    base_y = img.shape[0] - 24
    larg = img.shape[1]
    if not nomes:
        _vidro(img, larg // 2 - 70, base_y - 32, larg // 2 + 70, base_y + 12, 0.5, 12)
        texto(img, "EM PE", larg // 2, base_y, 0.8, cfg.COR_ROSA_CLARO, 2, centro=True)
        return
    msg = " + ".join(nomes)
    (w, _), _ = cv2.getTextSize(msg, FONTE, 0.9, 2)
    extra = 46 if a.inclinacao else 0
    x0 = larg // 2 - (w + extra) // 2 - 14
    _vidro(img, x0, base_y - 36, x0 + w + extra + 28, base_y + 14, 0.6, 14)
    texto(img, msg, x0 + 14 + (extra if a.inclinacao < 0 else 0), base_y, 0.9,
          cfg.COR_BRANCO, 2, brilho=True)
    if a.inclinacao:   # seta indicando o lado da inclinação
        yc = base_y - 10
        if a.inclinacao < 0:
            cv2.arrowedLine(img, (x0 + 42, yc), (x0 + 12, yc), cfg.COR_CIANO, 3, tipLength=0.5)
        else:
            cv2.arrowedLine(img, (x0 + w + 22, yc), (x0 + w + 52, yc), cfg.COR_CIANO, 3,
                            tipLength=0.5)


# ==========================================================================
# HUD
# ==========================================================================
def desenhar_hud(img: np.ndarray, jogo: Jogo, fps: float) -> None:
    larg, alt = img.shape[1], img.shape[0]
    _vidro(img, 4, 4, larg - 5, 64, 0.5, 16)
    texto(img, f"PONTOS {jogo.pontos}", 16, 30, 0.8, cfg.COR_BRANCO, 2, brilho=True)
    for i in range(cfg.VIDAS_INICIAIS):   # vidas: corações (vazios = perdidos)
        _coracao(img, 26 + i * 30, 50, 10, cfg.COR_ROSA if i < jogo.vidas else cfg.COR_CINZA,
                 cheio=i < jogo.vidas)
    for i in range(jogo.escudos):         # escudos: gemas
        _diamante(img, 126 + i * 26, 48, 9)
    texto(img, f"FASE {jogo.fase}", larg - 16, 28, 0.8, cfg.COR_DOURADO, 2, direita=True,
          brilho=True)
    bx1, bx2 = larg - 160, larg - 16      # barra de progresso da fase
    _retangulo_arredondado(img, bx1, 38, bx2, 48, cfg.COR_AMEIXA, -1, 5)
    preenchido = int((bx2 - bx1) * jogo.progresso_fase)
    if preenchido > 8:
        _retangulo_arredondado(img, bx1, 38, bx1 + preenchido, 48, cfg.COR_DOURADO, -1, 5)
    _retangulo_arredondado(img, bx1, 38, bx2, 48, cfg.COR_BRANCO, 1, 5)
    if jogo.combo > 0:
        cor = cfg.COR_DOURADO if jogo.multiplicador > 1 else cfg.COR_BRANCO
        texto(img, f"COMBO {jogo.combo}  x{jogo.multiplicador}", larg // 2, 30, 0.7,
              cor, 2, centro=True, brilho=jogo.multiplicador > 1)
    texto(img, f"FPS {fps:4.1f}", 10, alt - 8, 0.45, cfg.COR_ROSA_CLARO)
    texto(img, f"RECORDE {jogo.recorde}", larg - 10, alt - 8, 0.45, cfg.COR_ROSA_CLARO,
          direita=True)
    if jogo.banner_fase > 0:
        _vidro(img, 20, 140, larg - 20, 240, 0.6, 24)
        texto(img, f"FASE {jogo.fase}!", larg // 2, 205, 2.0, cfg.COR_BRANCO, 3, centro=True,
              fonte=FONTE_SCRIPT, brilho=True)
        texto(img, f"+{cfg.BONUS_FASE} pontos | mais rapido!", larg // 2, 230, 0.5,
              cfg.COR_ROSA_CLARO, centro=True)
        t = _agora()
        for i in range(6):
            _estrela(img, 60 + i * 104, 160 + 12 * math.sin(t * 5 + i), 8, cfg.COR_DOURADO)


def desenhar_debug(img: np.ndarray, jogo: Jogo, leitura: Optional[Leitura],
                   corpo: Optional[Corpo]) -> None:
    """Modo debug (tecla D): landmarks, visibilidade e valores dos limiares."""
    if leitura is not None and leitura.pontos is not None:
        for i, (x, y, v) in enumerate(leitura.pontos):
            cor = cfg.COR_VERDE if v >= cfg.LIMIAR_VISIBILIDADE else cfg.COR_VERMELHO
            cv2.circle(img, (int(x), int(y)), 3, cor, -1)
            cv2.putText(img, str(i), (int(x) + 4, int(y) - 3), cv2.FONT_HERSHEY_PLAIN,
                        0.8, cor, 1, cv2.LINE_AA)
    linhas = [f"estado: {jogo.estado.name}", f"vel: {jogo.velocidade:.0f}px/s  obst: {len(jogo.obstaculos)}"]
    if jogo.calibracao is not None:
        linhas.append(f"L (tronco): {jogo.calibracao.tronco:.0f}px")
    if corpo is not None:
        a = corpo.acoes
        joelho = f"{a.angulo_joelho:.0f}" if a.angulo_joelho is not None else "--"
        linhas += [
            f"queda ombro: {a.queda_ombro:+.2f}L (>={cfg.AGACHAR_ENTRA_FRAC})",
            f"subida quad: {a.subida_quadril:+.2f}L (>={cfg.PULAR_ENTRA_FRAC})",
            f"incl tronco: {a.angulo_tronco:+.1f}deg (>={cfg.INCLINAR_ENTRA_GRAUS:.0f})",
            f"angulo joelho: {joelho}deg (<{cfg.JOELHO_AGACHAR_GRAUS:.0f})",
        ]
    if leitura is not None and leitura.pontos is not None:
        p = leitura.pontos
        linhas.append("vis ombros %.2f/%.2f quad %.2f/%.2f" % (p[11][2], p[12][2], p[23][2], p[24][2]))
        linhas.append("vis nariz %.2f  punhos %.2f/%.2f" % (p[0][2], p[15][2], p[16][2]))
    larg = img.shape[1]
    _vidro(img, larg - 296, 70, larg - 6, 70 + 18 * len(linhas) + 12, 0.65, 10)
    for i, ln in enumerate(linhas):
        texto(img, ln, larg - 288, 90 + 18 * i, 0.4, cfg.COR_VERDE)


# ==========================================================================
# Telas de estado
# ==========================================================================
def _titulo_painel(img: np.ndarray, titulo: str, sub: str = "", cor: Cor = cfg.COR_BRANCO,
                   y: int = 200) -> None:
    larg = img.shape[1]
    _vidro(img, 20, y - 62, larg - 20, y + 56, 0.62, 22)
    texto(img, titulo, larg // 2, y, 1.1, cor, 2, centro=True, brilho=True)
    if sub:
        texto(img, sub, larg // 2, y + 36, 0.55, cfg.COR_ROSA_CLARO, centro=True)


def desenhar_estado(img: np.ndarray, jogo: Jogo) -> None:
    larg, alt = img.shape[1], img.shape[0]
    e = jogo.estado
    t = _agora()
    if e == Estado.MENU:
        _vidro(img, 18, 18, larg - 18, alt - 18, 0.58, 28)
        texto(img, "Esquiva Glam", larg // 2, 110, 2.6, cfg.COR_BRANCO, 4, centro=True,
              fonte=FONTE_SCRIPT, brilho=True)
        texto(img, "~ pink tech edition ~", larg // 2, 142, 0.6, cfg.COR_CIANO, 1, centro=True)
        regras = ["Desvie dos obstaculos usando o corpo!",
                  "Barra pink: AGACHE   |   Caixa glam: PULE",
                  "Cristal lateral: INCLINE o tronco para o lado oposto",
                  "Toque moedas e gemas com as maos",
                  "Raspar no obstaculo da bonus. Combos multiplicam pontos."]
        for i, r in enumerate(regras):
            texto(img, r, larg // 2, 190 + 28 * i, 0.55, cfg.COR_BRANCO, centro=True)
        pulso = 0.7 + 0.1 * math.sin(t * 4)
        texto(img, "Pressione ESPACO para calibrar e comecar", larg // 2, 365, pulso,
              cfg.COR_DOURADO, 2, centro=True, brilho=True)
        texto(img, "R reiniciar | C recalibrar | D debug | F tela cheia | Q sair", larg // 2, 405,
              0.45, cfg.COR_ROSA_CLARO, centro=True)
        for i in range(8):
            _estrela(img, 50 + i * 77, 435 + 8 * math.sin(t * 3 + i), 7 + 3 * math.sin(t * 5 + i),
                     cfg.COR_ROSA_CLARO)
    elif e == Estado.CALIBRANDO:
        _vidro(img, 10, 8, larg - 10, 112, 0.62, 18)
        texto(img, "CALIBRACAO", larg // 2, 40, 0.9, cfg.COR_DOURADO, 2, centro=True, brilho=True)
        texto(img, jogo.calib_mensagem, larg // 2, 68, 0.55, cfg.COR_BRANCO, centro=True)
        x1, x2 = larg // 2 - 150, larg // 2 + 150
        _retangulo_arredondado(img, x1, 80, x2, 100, cfg.COR_AMEIXA, -1, 8)
        cheio = int(300 * jogo.calib_progresso)
        if cheio > 10:
            _retangulo_arredondado(img, x1, 80, x1 + cheio, 100, cfg.COR_CIANO, -1, 8)
        _retangulo_arredondado(img, x1, 80, x2, 100, cfg.COR_BRANCO, 1, 8)
        texto(img, "Fique em pe, de frente, bracos relaxados", larg // 2, alt - 20, 0.55,
              cfg.COR_BRANCO, centro=True)
    elif e == Estado.CONTAGEM:
        n = max(1, math.ceil(jogo.contagem_restante))
        sub = "Retomando..." if jogo.contagem_retomada else "Prepare-se!"
        frac = jogo.contagem_restante - math.floor(jogo.contagem_restante)
        _vidro(img, 160, 140, larg - 160, 340, 0.5, 40)
        texto(img, str(n), larg // 2, 285, 4.0 + 0.8 * frac, cfg.COR_BRANCO, 6, centro=True,
              brilho=True)
        texto(img, sub, larg // 2, 322, 0.7, cfg.COR_ROSA_CLARO, 2, centro=True)
        for i in range(8):
            ang = t * 2 + i * math.pi / 4
            _estrela(img, larg // 2 + 110 * math.cos(ang), 235 + 70 * math.sin(ang), 8,
                     cfg.COR_DOURADO)
    elif e == Estado.PAUSADO:
        if jogo.motivo_pausa == MotivoPausa.PERDIDO:
            cv2.rectangle(img, (0, 0), (larg - 1, alt - 1), cfg.COR_VERMELHO, 8)
            _titulo_painel(img, "Volte para a frente da camera",
                           "Jogo pausado. Retoma com contagem quando voce voltar.", cfg.COR_BRANCO)
        else:
            _titulo_painel(img, "PAUSADO", "ESPACO para continuar | R reiniciar", cfg.COR_DOURADO)
    elif e == Estado.GAME_OVER:
        _vidro(img, 18, 18, larg - 18, alt - 18, 0.62, 28)
        texto(img, "Game Over", larg // 2, 135, 2.8, cfg.COR_BRANCO, 4, centro=True,
              fonte=FONTE_SCRIPT, brilho=True)
        novo = jogo.pontos >= jogo.recorde and jogo.pontos > 0
        texto(img, f"Pontos: {jogo.pontos}" + ("  NOVO RECORDE!" if novo else ""),
              larg // 2, 205, 0.9, cfg.COR_DOURADO, 2, centro=True, brilho=True)
        stats = [f"Recorde: {jogo.recorde}", f"Fase alcancada: {jogo.fase}",
                 f"Esquivas: {jogo.esquivas}   Moedas: {jogo.moedas}",
                 f"Melhor combo: {jogo.melhor_combo}   Tempo: {jogo.tempo:.0f}s"]
        for i, s in enumerate(stats):
            texto(img, s, larg // 2, 250 + 30 * i, 0.65, cfg.COR_BRANCO, centro=True)
        texto(img, "ESPACO ou R para jogar de novo", larg // 2, 410, 0.7, cfg.COR_CIANO, 2,
              centro=True, brilho=True)


# ==========================================================================
# Composição da cena inteira
# ==========================================================================
def desenhar_cena(img: np.ndarray, jogo: Jogo, leitura: Optional[Leitura],
                  corpo: Optional[Corpo], efeitos: Efeitos, fps: float, debug: bool) -> None:
    """Desenha, em ordem: fundo, glitter, zonas, objetos, corpo, efeitos, HUD e mensagens."""
    aplicar_fundo(img)
    efeitos.desenhar_ambiente(img)
    if jogo.calibracao is not None and jogo.estado not in (Estado.MENU, Estado.CALIBRANDO):
        desenhar_zonas(img, jogo.calibracao, debug)
    for it in jogo.itens:
        desenhar_item(img, it, jogo.tempo)
    for o in jogo.obstaculos:
        desenhar_obstaculo(img, o, jogo.tempo)
    if leitura is not None:
        desenhar_esqueleto(img, leitura)
    if corpo is not None and jogo.estado not in (Estado.MENU, Estado.CALIBRANDO):
        desenhar_caixas(img, corpo, jogo.invulneravel, jogo.tempo)
        desenhar_acoes(img, corpo)
    if debug:
        desenhar_debug(img, jogo, leitura, corpo)
    efeitos.desenhar(img)
    if jogo.estado not in (Estado.MENU, Estado.CALIBRANDO):
        desenhar_hud(img, jogo, fps)
    desenhar_estado(img, jogo)
