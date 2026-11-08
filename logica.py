"""Regras do jogo "Esquiva com o corpo".

Este módulo NÃO importa OpenCV nem MediaPipe: ele só recebe um `Entrada`
(corpo com caixas de colisão e mãos, em pixels) e evolui o estado do jogo.
Assim dá para trocar regras sem mexer na visão e testar sem câmera.

Máquina de estados:

    MENU ──ESPAÇO──> CALIBRANDO ──calibrou──> CONTAGEM ──3,2,1──> JOGANDO
                                                  ^                  │  │
                                  (volta a pessoa / ESPAÇO)          │  └─ vidas = 0 ─> GAME_OVER
                                                  │                  v
                                               PAUSADO <──── ESPAÇO ou pessoa perdida
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import config as cfg
from modelos import Calibracao, Corpo, Entrada, Retangulo


class Estado(Enum):
    MENU = auto()
    CALIBRANDO = auto()
    CONTAGEM = auto()
    JOGANDO = auto()
    PAUSADO = auto()
    GAME_OVER = auto()


class MotivoPausa(Enum):
    NENHUM = auto()
    MANUAL = auto()
    PERDIDO = auto()     # jogador saiu do quadro


class TipoObstaculo(Enum):
    ALTO = auto()        # barra suspensa: exige AGACHAR
    BAIXO = auto()       # bloco no chão: exige PULAR
    LATERAL = auto()     # pedra que cai ao lado: exige INCLINAR para o lado oposto


class TipoItem(Enum):
    MOEDA = auto()
    ESCUDO = auto()


class TipoEvento(Enum):
    """Eventos que a lógica emite para o visual/som reagirem."""
    INICIO = auto()
    CONTAGEM = auto()
    DANO = auto()
    ESCUDO_USADO = auto()
    MOEDA = auto()
    ESCUDO = auto()
    ESQUIVA = auto()
    RASPAO = auto()
    COMBO_PERDIDO = auto()
    FASE = auto()
    GAME_OVER = auto()


@dataclass
class Evento:
    tipo: TipoEvento
    x: float = 0.0
    y: float = 0.0
    valor: int = 0


@dataclass
class Obstaculo:
    tipo: TipoObstaculo
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    aviso: float = 0.0           # >0: pedra ainda só avisando (sem colisão)
    lado: int = 0                # LATERAL: -1 esquerda, +1 direita da tela
    atingido: bool = False
    contabilizado: bool = False
    folga_min: float = math.inf  # menor distância já registrada até o corpo (px)

    @property
    def ret(self) -> Retangulo:
        return Retangulo(self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Item:
    tipo: TipoItem
    x: float
    y: float
    raio: float = cfg.ITEM_RAIO_PX
    vx: float = 0.0


class Jogo:
    """Estado completo da partida + regras de atualização."""

    def __init__(self, largura: int = cfg.LARGURA, altura: int = cfg.ALTURA,
                 semente: Optional[int] = None) -> None:
        self.largura = largura
        self.altura = altura
        self.rng = random.Random(semente)
        self.estado = Estado.MENU
        self.motivo_pausa = MotivoPausa.NENHUM
        self.calibracao: Optional[Calibracao] = None
        self.recorde = 0
        self.calib_progresso = 0.0
        self.calib_mensagem = ""
        self.contagem_restante = 0.0
        self.contagem_retomada = False   # True quando é retomada (não partida nova)
        self._sem_pessoa = 0.0
        self._com_pessoa = 0.0
        self._eventos: List[Evento] = []
        self._resetar_partida()

    # ------------------------------------------------------------------
    # Comandos vindos do teclado
    # ------------------------------------------------------------------
    def alternar_espaco(self) -> None:
        """ESPAÇO: iniciar / pausar / retomar / jogar de novo."""
        if self.estado == Estado.MENU:
            self._ir_calibrar()
        elif self.estado == Estado.JOGANDO:
            self.estado = Estado.PAUSADO
            self.motivo_pausa = MotivoPausa.MANUAL
        elif self.estado == Estado.PAUSADO and self.motivo_pausa == MotivoPausa.MANUAL:
            self._iniciar_contagem(retomada=True)
        elif self.estado == Estado.GAME_OVER:
            self.reiniciar()

    def reiniciar(self) -> None:
        """R: nova partida (mantém a calibração, se existir)."""
        if self.estado == Estado.MENU:
            self._ir_calibrar()
        elif self.calibracao is None:
            self._ir_calibrar()
        else:
            self._resetar_partida()
            self._iniciar_contagem(retomada=False)

    def recalibrar(self) -> None:
        """C: descarta a calibração e mede o jogador de novo."""
        self._ir_calibrar()

    def definir_calibracao(self, calibracao: Calibracao) -> None:
        """Chamado pelo main quando o Calibrador termina."""
        self.calibracao = calibracao
        if self.estado == Estado.CALIBRANDO:
            self._resetar_partida()
            self._iniciar_contagem(retomada=False)

    def avancar_fase(self) -> None:
        """Atalho de depuração (tecla N): pula para a próxima fase."""
        if self.estado == Estado.JOGANDO:
            self.tempo = self.fase * cfg.FASE_DURACAO_S
            self._verificar_fase()

    def coletar_eventos(self) -> List[Evento]:
        eventos, self._eventos = self._eventos, []
        return eventos

    # ------------------------------------------------------------------
    # Atualização por quadro
    # ------------------------------------------------------------------
    def atualizar(self, dt: float, entrada: Entrada) -> None:
        self.calib_progresso = entrada.calib_progresso
        self.calib_mensagem = entrada.calib_mensagem

        if self.estado in (Estado.MENU, Estado.CALIBRANDO, Estado.GAME_OVER):
            return

        # --- detecção perdida: pausa automática e retomada com contagem ---
        if entrada.presente and entrada.corpo is not None:
            self._sem_pessoa = 0.0
            self._com_pessoa += dt
        else:
            self._sem_pessoa += dt
            self._com_pessoa = 0.0

        perdida = self._sem_pessoa > cfg.TOLERANCIA_PERDA_S
        if self.estado in (Estado.JOGANDO, Estado.CONTAGEM) and perdida:
            self.estado = Estado.PAUSADO
            self.motivo_pausa = MotivoPausa.PERDIDO
            return
        if (self.estado == Estado.PAUSADO and self.motivo_pausa == MotivoPausa.PERDIDO
                and self._com_pessoa >= cfg.REGRESSO_ESTAVEL_S):
            self._iniciar_contagem(retomada=True)
            return

        if self.estado == Estado.CONTAGEM:
            antes = math.ceil(self.contagem_restante)
            self.contagem_restante -= dt
            if self.contagem_restante <= 0:
                self.estado = Estado.JOGANDO
                self._emitir(TipoEvento.INICIO)
            elif math.ceil(self.contagem_restante) != antes:
                self._emitir(TipoEvento.CONTAGEM, valor=math.ceil(self.contagem_restante))
        elif self.estado == Estado.JOGANDO and entrada.corpo is not None:
            self._atualizar_jogo(dt, entrada.corpo)

    # ------------------------------------------------------------------
    # Propriedades de dificuldade
    # ------------------------------------------------------------------
    @property
    def multiplicador(self) -> int:
        return min(cfg.MULT_MAX, 1 + self.combo // cfg.COMBO_POR_NIVEL)

    @property
    def velocidade(self) -> float:
        return min(cfg.VEL_MAX, cfg.VEL_BASE + (self.fase - 1) * cfg.VEL_PASSO)

    @property
    def intervalo_obstaculos(self) -> float:
        return max(cfg.INTERVALO_MIN_S, cfg.INTERVALO_BASE_S - (self.fase - 1) * cfg.INTERVALO_PASSO_S)

    @property
    def gap_chegada(self) -> float:
        return max(cfg.GAP_MIN_S, cfg.GAP_BASE_S - (self.fase - 1) * cfg.GAP_PASSO_S)

    @property
    def progresso_fase(self) -> float:
        return (self.tempo % cfg.FASE_DURACAO_S) / cfg.FASE_DURACAO_S

    # ------------------------------------------------------------------
    # Transições internas
    # ------------------------------------------------------------------
    def _resetar_partida(self) -> None:
        self.pontos = 0
        self._pontos_frac = 0.0
        self.vidas = cfg.VIDAS_INICIAIS
        self.escudos = 0
        self.fase = 1
        self.combo = 0
        self.melhor_combo = 0
        self.esquivas = 0
        self.moedas = 0
        self.tempo = 0.0
        self.invulneravel = 0.0
        self.obstaculos: List[Obstaculo] = []
        self.itens: List[Item] = []
        self.banner_fase = 0.0
        self._timer_obst = 1.2
        self._timer_item = 2.5
        self._ultima_chegada = -math.inf
        self._tipo_pendente: Optional[TipoObstaculo] = None

    def _ir_calibrar(self) -> None:
        self.estado = Estado.CALIBRANDO
        self.motivo_pausa = MotivoPausa.NENHUM
        self.calibracao = None
        self.calib_progresso = 0.0
        self._resetar_partida()

    def _iniciar_contagem(self, retomada: bool) -> None:
        self.estado = Estado.CONTAGEM
        self.motivo_pausa = MotivoPausa.NENHUM
        self.contagem_restante = float(cfg.CONTAGEM_S)
        self.contagem_retomada = retomada
        self._sem_pessoa = 0.0
        self._emitir(TipoEvento.CONTAGEM, valor=cfg.CONTAGEM_S)

    def _emitir(self, tipo: TipoEvento, x: float = 0.0, y: float = 0.0, valor: int = 0) -> None:
        self._eventos.append(Evento(tipo, x, y, valor))

    # ------------------------------------------------------------------
    # Loop do jogo (estado JOGANDO)
    # ------------------------------------------------------------------
    def _atualizar_jogo(self, dt: float, corpo: Corpo) -> None:
        self.tempo += dt
        self.invulneravel = max(0.0, self.invulneravel - dt)
        self.banner_fase = max(0.0, self.banner_fase - dt)

        # pontuação por tempo de sobrevivência
        self._pontos_frac += cfg.PONTOS_POR_SEGUNDO * self.multiplicador * dt
        inteiros = int(self._pontos_frac)
        self.pontos += inteiros
        self._pontos_frac -= inteiros

        self._verificar_fase()
        self._gerar(dt)
        self._mover(dt)
        self._colisoes(corpo)
        self._coletas(corpo)
        self._limpar(corpo)

    def _verificar_fase(self) -> None:
        nova = 1 + int(self.tempo // cfg.FASE_DURACAO_S)
        if nova > self.fase:
            self.fase = nova
            self.pontos += cfg.BONUS_FASE
            self.banner_fase = 2.2
            self._emitir(TipoEvento.FASE, valor=self.fase)

    # --- geração ---------------------------------------------------------
    def _sortear_tipo(self) -> TipoObstaculo:
        tipos = [TipoObstaculo.ALTO, TipoObstaculo.BAIXO]
        if self.fase >= cfg.FASE_LATERAL:
            tipos.append(TipoObstaculo.LATERAL)
        return self.rng.choice(tipos)

    def _velocidade_queda(self) -> float:
        return max(380.0, 1.7 * self.velocidade)

    def _tempo_chegada(self, tipo: TipoObstaculo) -> float:
        """Segundos entre gerar o obstáculo e ele alcançar o jogador."""
        c = self.calibracao
        if tipo == TipoObstaculo.LATERAL:
            h = cfg.LATERAL_ALTURA_FRAC * c.tronco
            dist = c.ombro_y - 0.7 * c.tronco + h
            return self._aviso_lateral() + max(dist, 0.0) / self._velocidade_queda()
        dist = (self.largura + 10) - (c.centro_x + 0.3 * c.tronco)
        return dist / self.velocidade

    def _aviso_lateral(self) -> float:
        return max(0.5, cfg.LATERAL_AVISO_S - 0.05 * (self.fase - 1))

    def _gerar(self, dt: float) -> None:
        c = self.calibracao
        # Obstáculos: respeita intervalo e uma separação mínima entre CHEGADAS
        # ao jogador (assim barras e pedras nunca exigem duas ações ao mesmo tempo).
        self._timer_obst -= dt
        if self._timer_obst <= 0:
            tipo = self._tipo_pendente or self._sortear_tipo()
            self._tipo_pendente = tipo
            chegada = self.tempo + self._tempo_chegada(tipo)
            if chegada >= self._ultima_chegada + self.gap_chegada:
                self.obstaculos.append(self._criar_obstaculo(tipo))
                self._ultima_chegada = chegada
                self._tipo_pendente = None
                self._timer_obst = self.intervalo_obstaculos * self.rng.uniform(0.85, 1.25)

        # Itens coletáveis (moeda / escudo) numa altura alcançável com o braço
        self._timer_item -= dt
        if self._timer_item <= 0:
            livre = all(o.tipo == TipoObstaculo.LATERAL or o.x < self.largura - 200
                        for o in self.obstaculos)
            if livre:
                escudo = (self.escudos < cfg.ESCUDOS_MAX
                          and self.rng.random() < cfg.ITEM_CHANCE_ESCUDO)
                y_min = max(50.0, c.ombro_y - 1.1 * c.tronco)
                y_max = min(self.altura - 50.0, c.quadril_y)
                self.itens.append(Item(
                    TipoItem.ESCUDO if escudo else TipoItem.MOEDA,
                    x=self.largura + 30.0,
                    y=self.rng.uniform(y_min, max(y_min + 1, y_max)),
                    vx=-self.velocidade))
                self._timer_item = self.rng.uniform(*cfg.ITEM_INTERVALO_S)
            else:
                self._timer_item = 0.3

    def _criar_obstaculo(self, tipo: TipoObstaculo) -> Obstaculo:
        """Cria o obstáculo com a geometria definida pela CALIBRAÇÃO (relativa ao corpo)."""
        c = self.calibracao
        L = c.tronco
        if tipo == TipoObstaculo.ALTO:
            # barra do teto até 0.45L acima dos ombros: a cabeça em pé bate, agachado passa
            base = c.ombro_y - cfg.ALTO_FOLGA_FRAC * L
            return Obstaculo(tipo, self.largura + 10.0, 0.0, cfg.OBST_LARGURA_PX, base,
                             vx=-self.velocidade)
        if tipo == TipoObstaculo.BAIXO:
            # bloco do chão até quase a altura do quadril: só passa pulando
            topo = c.quadril_y - cfg.BAIXO_TOPO_FRAC * L
            return Obstaculo(tipo, self.largura + 10.0, topo, cfg.OBST_LARGURA_PX,
                             self.altura - topo, vx=-self.velocidade)
        # LATERAL: pedra que cai cobrindo uma das metades do corpo
        lado = self.rng.choice((-1, 1))
        w = cfg.LATERAL_LARGURA_FRAC * L
        h = cfg.LATERAL_ALTURA_FRAC * L
        borda = cfg.LATERAL_BORDA_FRAC * L
        x = c.centro_x + borda if lado > 0 else c.centro_x - borda - w
        return Obstaculo(tipo, x, -h, w, h, vy=self._velocidade_queda(),
                         aviso=self._aviso_lateral(), lado=lado)

    # --- movimento / colisão / coleta ------------------------------------
    def _mover(self, dt: float) -> None:
        for o in self.obstaculos:
            if o.aviso > 0:
                o.aviso -= dt
                continue
            o.x += o.vx * dt
            o.y += o.vy * dt
        for it in self.itens:
            it.x += it.vx * dt

    def _colisoes(self, corpo: Corpo) -> None:
        for o in self.obstaculos:
            if o.aviso > 0 or o.atingido:
                continue
            if o.tipo == TipoObstaculo.BAIXO and corpo.acoes.pulando:
                continue   # pulo registrado = no ar: passa por cima do bloco baixo
            r = o.ret
            folga = min(r.distancia(corpo.cabeca), r.distancia(corpo.tronco))
            o.folga_min = min(o.folga_min, folga)
            if folga == 0.0:
                o.atingido = True          # cada obstáculo só causa dano uma vez
                if self.invulneravel <= 0:
                    cx, cy = r.centro
                    self._aplicar_dano(cx, cy)
                    if self.estado == Estado.GAME_OVER:
                        return

    def _aplicar_dano(self, x: float, y: float) -> None:
        if self.combo >= 3:
            self._emitir(TipoEvento.COMBO_PERDIDO, x, y)
        self.combo = 0
        if self.escudos > 0:
            self.escudos -= 1
            self.invulneravel = cfg.INVULNERAVEL_ESCUDO_S
            self._emitir(TipoEvento.ESCUDO_USADO, x, y)
            return
        self.vidas -= 1
        self.invulneravel = cfg.INVULNERAVEL_DANO_S
        self._emitir(TipoEvento.DANO, x, y, self.vidas)
        if self.vidas <= 0:
            self.recorde = max(self.recorde, self.pontos)
            self.estado = Estado.GAME_OVER
            self._emitir(TipoEvento.GAME_OVER, x, y, self.pontos)

    def _passou(self, o: Obstaculo, corpo: Corpo) -> bool:
        """O obstáculo já ultrapassou o jogador sem encostar?"""
        if o.tipo == TipoObstaculo.LATERAL:
            return o.y > corpo.tronco.y2
        return o.x + o.w < corpo.tronco.x1 - 4

    def _registrar_acerto(self, pontos_base: int) -> int:
        """Soma pontos × multiplicador, sobe o combo e devolve os pontos ganhos."""
        self.combo += 1
        self.melhor_combo = max(self.melhor_combo, self.combo)
        ganho = pontos_base * self.multiplicador
        self.pontos += ganho
        return ganho

    def _limpar(self, corpo: Corpo) -> None:
        L = self.calibracao.tronco
        restantes: List[Obstaculo] = []
        for o in self.obstaculos:
            if (not o.atingido and not o.contabilizado and o.aviso <= 0
                    and self._passou(o, corpo)):
                o.contabilizado = True
                self.esquivas += 1
                cx, cy = corpo.tronco.centro
                ganho = self._registrar_acerto(cfg.PONTOS_ESQUIVA)
                self._emitir(TipoEvento.ESQUIVA, cx, corpo.cabeca.y1, ganho)
                if o.folga_min < cfg.RASPAO_FRAC * L:
                    bonus = cfg.PONTOS_RASPAO * self.multiplicador
                    self.pontos += bonus
                    self._emitir(TipoEvento.RASPAO, cx, corpo.cabeca.y1 - 28, bonus)
            fora = (o.x + o.w < -20) or (o.y > self.altura + 20)
            if not fora:
                restantes.append(o)
        self.obstaculos = restantes
        self.itens = [i for i in self.itens if i.x > -40]

    def _coletas(self, corpo: Corpo) -> None:
        restantes: List[Item] = []
        for it in self.itens:
            pego = any(math.hypot(mx - it.x, my - it.y) <= it.raio + cfg.MAO_RAIO_PX
                       for mx, my in corpo.maos)
            if not pego:
                restantes.append(it)
                continue
            if it.tipo == TipoItem.MOEDA:
                self.moedas += 1
                ganho = self._registrar_acerto(cfg.PONTOS_MOEDA)
                self._emitir(TipoEvento.MOEDA, it.x, it.y, ganho)
            else:
                self.escudos = min(cfg.ESCUDOS_MAX, self.escudos + 1)
                ganho = self._registrar_acerto(cfg.PONTOS_ESCUDO)
                self._emitir(TipoEvento.ESCUDO, it.x, it.y, ganho)
        self.itens = restantes
