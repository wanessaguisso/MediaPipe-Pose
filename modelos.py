"""Tipos de dados compartilhados entre pose.py (visão), logica.py (regras)
e visual.py (desenho). Não depende de OpenCV nem de MediaPipe.

Todas as coordenadas estão em pixels do quadro (LARGURA x ALTURA do config).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Retangulo:
    """Caixa alinhada aos eixos: (x1, y1) canto superior esquerdo."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centro(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def distancia(self, outro: "Retangulo") -> float:
        """Folga entre dois retângulos em pixels; 0 se eles se sobrepõem."""
        dx = max(0.0, outro.x1 - self.x2, self.x1 - outro.x2)
        dy = max(0.0, outro.y1 - self.y2, self.y1 - outro.y2)
        return math.hypot(dx, dy)

    def intersecta(self, outro: "Retangulo") -> bool:
        return self.distancia(outro) == 0.0

    def distancia_ponto(self, x: float, y: float) -> float:
        dx = max(self.x1 - x, 0.0, x - self.x2)
        dy = max(self.y1 - y, 0.0, y - self.y2)
        return math.hypot(dx, dy)


@dataclass
class Calibracao:
    """Medidas do jogador em pé, feitas na calibração."""
    centro_x: float       # x médio do tronco
    ombro_y: float        # y médio dos ombros
    quadril_y: float      # y médio dos quadris
    nariz_y: float
    tronco: float         # L = quadril_y - ombro_y (unidade das zonas)
    largura_ombros: float


@dataclass
class Acoes:
    """Ações reconhecidas a partir da pose + métricas numéricas (debug)."""
    agachado: bool = False
    pulando: bool = False
    inclinacao: int = 0              # -1 = esquerda da tela, +1 = direita, 0 = reto
    mao_esq_alta: bool = False
    mao_dir_alta: bool = False
    queda_ombro: float = 0.0         # em L (positivo = ombros desceram)
    subida_quadril: float = 0.0      # em L (positivo = quadril subiu)
    angulo_tronco: float = 0.0       # graus; positivo = inclinado p/ direita da tela
    angulo_joelho: Optional[float] = None


@dataclass
class Corpo:
    """O que o jogo precisa saber do jogador a cada quadro."""
    cabeca: Retangulo
    tronco: Retangulo
    maos: List[Tuple[float, float]] = field(default_factory=list)
    acoes: Acoes = field(default_factory=Acoes)


@dataclass
class Entrada:
    """Resumo da visão para a lógica do jogo, a cada quadro."""
    presente: bool                   # corpo confiável dentro do quadro?
    corpo: Optional[Corpo] = None    # None se não há calibração ou pessoa
    calib_progresso: float = 0.0     # 0..1 durante a calibração
    calib_mensagem: str = ""
