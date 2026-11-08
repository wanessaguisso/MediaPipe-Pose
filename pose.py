"""Visão computacional: captura, MediaPipe PoseLandmarker, suavização,
calibração e tradução dos landmarks em AÇÕES (agachar, pular, inclinar...).

Usa a API MediaPipe Tasks (PoseLandmarker em modo VIDEO), nunca `mp.solutions`.
Tudo aqui trabalha em pixels do quadro LARGURA x ALTURA (config.py).
"""
from __future__ import annotations

import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from modelos import Acoes, Calibracao, Corpo, Retangulo

# Índices dos landmarks do modelo BlazePose (33 pontos).
# Fonte: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
NARIZ = 0
OMBRO_E, OMBRO_D = 11, 12
PUNHO_E, PUNHO_D = 15, 16
INDICE_E, INDICE_D = 19, 20
QUADRIL_E, QUADRIL_D = 23, 24
JOELHO_E, JOELHO_D = 25, 26
TORNOZELO_E, TORNOZELO_D = 27, 28
OBRIGATORIOS = (OMBRO_E, OMBRO_D, QUADRIL_E, QUADRIL_D)

# Um ponto suavizado: (x_px, y_px, visibilidade 0..1)
Ponto = Tuple[float, float, float]


class ErroFonte(RuntimeError):
    """Câmera/vídeo não pôde ser aberto ou parou de entregar quadros."""


class ErroModelo(RuntimeError):
    """O arquivo pose_landmarker.task não existe e não pôde ser baixado."""


# ==========================================================================
# Modelo: download automático
# ==========================================================================
def garantir_modelo(caminho: Path = cfg.CAMINHO_MODELO) -> Path:
    """Baixa o pose_landmarker.task na primeira execução (urllib + progresso)."""
    if caminho.exists() and caminho.stat().st_size > 0:
        return caminho
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(".tmp")
    print(f"[modelo] Baixando {caminho.name} (primeira execução)...")
    print(f"[modelo] Origem: {cfg.URL_MODELO}")

    def progresso(blocos: int, tam_bloco: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, blocos * tam_bloco * 100.0 / total)
            print(f"\r[modelo] {pct:5.1f}% ({total / 1e6:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(cfg.URL_MODELO, temporario, reporthook=progresso)
        print()
    except (urllib.error.URLError, OSError, TimeoutError) as erro:
        temporario.unlink(missing_ok=True)
        raise ErroModelo(
            f"Não foi possível baixar o modelo ({erro}).\n"
            f"Verifique a internet ou baixe manualmente:\n  {cfg.URL_MODELO}\n"
            f"e salve em: {caminho}") from erro
    temporario.replace(caminho)
    print("[modelo] Download concluído.")
    return caminho


# ==========================================================================
# Fonte de vídeo (webcam ou arquivo)
# ==========================================================================
class FonteVideo:
    """Lê quadros já redimensionados para LARGURA x ALTURA.

    Webcam: espelhada (flip horizontal) para parecer um espelho.
    Arquivo: sem espelho; ao terminar reinicia (ou encerra com repetir=False).
    """

    def __init__(self, camera: int = 0, video: Optional[str] = None,
                 repetir: bool = True) -> None:
        self.eh_webcam = video is None
        self.repetir = repetir
        self._falhas = 0
        self._ultimo = time.perf_counter()
        if video is not None:
            caminho = Path(video)
            if not caminho.is_file():
                raise ErroFonte(f"Arquivo de vídeo não encontrado: {caminho}")
            self.cap = cv2.VideoCapture(str(caminho))
            self.nome = caminho.name
        else:
            # No Windows, CAP_DSHOW abre bem mais rápido que o backend padrão.
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(camera, backend)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(camera)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.LARGURA)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.ALTURA)
            self.cap.set(cv2.CAP_PROP_FPS, cfg.FPS_CAMERA)
            self.nome = f"camera {camera}"
        if not self.cap.isOpened():
            self.cap.release()
            raise ErroFonte(
                f"Não foi possível abrir {self.nome}. Feche outros apps que usam a "
                "câmera ou tente outro índice com --camera 1.")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._intervalo = 1.0 / fps if (not self.eh_webcam and 5 < fps < 120) else 0.0

    def ler(self) -> Optional[np.ndarray]:
        """Devolve um quadro BGR ou None quando a fonte acabou/falhou."""
        if self._intervalo:  # arquivo: reproduz em tempo real
            espera = self._intervalo - (time.perf_counter() - self._ultimo)
            if espera > 0:
                time.sleep(espera)
        self._ultimo = time.perf_counter()

        ok, quadro = self.cap.read()
        if not ok and not self.eh_webcam:
            if not self.repetir:
                return None                      # fim do vídeo: encerra limpo
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, quadro = self.cap.read()
        while not ok or quadro is None:          # webcam: tolera falhas passageiras
            self._falhas += 1
            if self._falhas > 30:                # ~1 s de falhas seguidas: desiste
                return None
            time.sleep(0.03)
            ok, quadro = self.cap.read()
        self._falhas = 0
        if quadro.shape[1] != cfg.LARGURA or quadro.shape[0] != cfg.ALTURA:
            quadro = cv2.resize(quadro, (cfg.LARGURA, cfg.ALTURA))
        if self.eh_webcam:
            quadro = cv2.flip(quadro, 1)
        return quadro

    def liberar(self) -> None:
        self.cap.release()


# ==========================================================================
# Detector + suavização
# ==========================================================================
@dataclass
class Leitura:
    """Resultado da visão em um quadro."""
    presente: bool                      # corpo confiável e dentro do quadro
    pontos: Optional[List[Ponto]]       # 33 pontos suavizados (None se não detectou)
    motivo: str = ""                    # explicação quando presente=False


class RastreadorPose:
    """Envolve o PoseLandmarker (Tasks API) e suaviza os landmarks."""

    def __init__(self, caminho_modelo: Path) -> None:
        import mediapipe as mp  # import tardio: mensagens de erro de CLI ficam rápidas
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        opcoes = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(caminho_modelo)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=cfg.CONF_DETECCAO,
            min_pose_presence_confidence=cfg.CONF_PRESENCA,
            min_tracking_confidence=cfg.CONF_RASTREIO,
        )
        self._detector = vision.PoseLandmarker.create_from_options(opcoes)
        self._ultimo_ts = -1
        self._xy: Optional[np.ndarray] = None       # (33, 2) suavizado
        self._vis = np.zeros(33)
        self._t_ultima_deteccao = 0.0

    def processar(self, quadro_bgr: np.ndarray) -> Leitura:
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        imagem = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # No modo VIDEO os timestamps precisam ser estritamente crescentes (ms).
        ts = max(int(time.perf_counter() * 1000), self._ultimo_ts + 1)
        self._ultimo_ts = ts
        resultado = self._detector.detect_for_video(imagem, ts)

        if not resultado.pose_landmarks:
            return Leitura(False, None, "nenhuma pessoa detectada")

        agora = time.perf_counter()
        lms = resultado.pose_landmarks[0]
        novo = np.array([[lm.x * cfg.LARGURA, lm.y * cfg.ALTURA] for lm in lms])
        # visibilidade efetiva = menor entre visibility e presence (quando existem)
        vis = np.array([min(_ou(lm.visibility, 1.0), _ou(lm.presence, 1.0)) for lm in lms])
        self._suavizar(novo, vis, agora)
        self._t_ultima_deteccao = agora

        pontos = [(float(x), float(y), float(v)) for (x, y), v in zip(self._xy, self._vis)]
        ok, motivo = _corpo_valido(pontos)
        return Leitura(ok, pontos, motivo)

    def _suavizar(self, novo: np.ndarray, vis: np.ndarray, agora: float) -> None:
        """Média móvel exponencial adaptativa, ponto a ponto.

        alfa pequeno => muito suave (sem tremor), alfa grande => acompanha rápido.
        O alfa de cada ponto cresce com o quanto ele se moveu no quadro.
        Pontos com baixa visibilidade NÃO atualizam a posição (mantém o último
        valor confiável em vez de "pular" para uma estimativa ruim).
        """
        reinicia = (self._xy is None
                    or agora - self._t_ultima_deteccao > cfg.REINICIO_FILTRO_S)
        if reinicia:
            self._xy = novo.copy()
        else:
            desloc = np.linalg.norm(novo - self._xy, axis=1)
            t = np.clip(desloc / cfg.ALFA_DIST_PX, 0.0, 1.0)
            alfa = (cfg.ALFA_MIN + (cfg.ALFA_MAX - cfg.ALFA_MIN) * t)[:, None]
            confiavel = (vis >= cfg.LIMIAR_VIS_OPCIONAL)[:, None]
            self._xy = np.where(confiavel, alfa * novo + (1 - alfa) * self._xy, self._xy)
        self._vis = vis

    def fechar(self) -> None:
        self._detector.close()


def _ou(valor: Optional[float], padrao: float) -> float:
    return padrao if valor is None else float(valor)


def _corpo_valido(pontos: List[Ponto]) -> Tuple[bool, str]:
    """Ombros e quadris precisam estar visíveis E dentro do quadro."""
    for i in OBRIGATORIOS:
        x, y, v = pontos[i]
        if v < cfg.LIMIAR_VISIBILIDADE:
            return False, "corpo parcialmente visivel"
        if not (0 <= x <= cfg.LARGURA) or y < 0 or y > cfg.ALTURA * cfg.MARGEM_BORDA_BAIXO:
            return False, "corpo fora do quadro"
    return True, ""


# ==========================================================================
# Medidas do corpo
# ==========================================================================
@dataclass
class Medidas:
    centro_x: float
    ombro_y: float
    quadril_y: float
    ombro_x: float
    quadril_x: float
    tronco: float
    largura_ombros: float
    nariz: Optional[Tuple[float, float]]


def _visivel(pontos: List[Ponto], i: int, limiar: float = cfg.LIMIAR_VIS_OPCIONAL) -> bool:
    return pontos[i][2] >= limiar


def medir(pontos: List[Ponto]) -> Medidas:
    """Extrai as medidas básicas do tronco a partir de ombros e quadris."""
    ombro_x = (pontos[OMBRO_E][0] + pontos[OMBRO_D][0]) / 2
    ombro_y = (pontos[OMBRO_E][1] + pontos[OMBRO_D][1]) / 2
    quadril_x = (pontos[QUADRIL_E][0] + pontos[QUADRIL_D][0]) / 2
    quadril_y = (pontos[QUADRIL_E][1] + pontos[QUADRIL_D][1]) / 2
    nariz = (pontos[NARIZ][0], pontos[NARIZ][1]) if _visivel(pontos, NARIZ) else None
    return Medidas(
        centro_x=(ombro_x + quadril_x) / 2,
        ombro_y=ombro_y, quadril_y=quadril_y,
        ombro_x=ombro_x, quadril_x=quadril_x,
        tronco=max(quadril_y - ombro_y, 1.0),
        largura_ombros=abs(pontos[OMBRO_E][0] - pontos[OMBRO_D][0]),
        nariz=nariz)


# ==========================================================================
# Calibração
# ==========================================================================
class Calibrador:
    """Mede o jogador em pé durante CALIBRACAO_SEG segundos.

    Só conta o tempo enquanto o corpo está estável, bem enquadrado e a uma
    distância razoável; qualquer problema zera o progresso e explica o motivo.
    """

    def __init__(self) -> None:
        self.mensagem = ""
        self.resetar()

    def resetar(self) -> None:
        self._amostras: List[Medidas] = []
        self._inicio: Optional[float] = None
        self.mensagem = "Fique em pe, de frente, mostrando da cabeca ao quadril"

    @property
    def pronto(self) -> bool:
        return (self._inicio is not None
                and time.perf_counter() - self._inicio >= cfg.CALIBRACAO_SEG
                and len(self._amostras) >= cfg.CALIB_MIN_AMOSTRAS)

    @property
    def progresso(self) -> float:
        if self._inicio is None:
            return 0.0
        return min(1.0, (time.perf_counter() - self._inicio) / cfg.CALIBRACAO_SEG)

    def alimentar(self, leitura: Leitura) -> float:
        """Processa um quadro e devolve o progresso (0..1)."""
        if not leitura.presente or leitura.pontos is None:
            self._reiniciar_com(f"Ajuste o enquadramento: {leitura.motivo}")
            return 0.0
        m = medir(leitura.pontos)
        if m.tronco < cfg.CALIB_TRONCO_MIN_PX:
            self._reiniciar_com("Chegue mais perto da camera")
            return 0.0
        if m.tronco > cfg.CALIB_TRONCO_MAX_FRAC * cfg.ALTURA:
            self._reiniciar_com("Afaste-se um pouco da camera")
            return 0.0
        if m.nariz is None or m.nariz[1] >= m.ombro_y:
            self._reiniciar_com("Enquadre a cabeca na imagem")
            return 0.0
        if self._amostras:
            ref = float(np.median([a.ombro_y for a in self._amostras]))
            if abs(m.ombro_y - ref) > cfg.CALIB_ESTABILIDADE_FRAC * m.tronco:
                self._reiniciar_com("Fique parado...")
                self._amostras = [m]       # recomeça a partir desta posição
                self._inicio = time.perf_counter()
                return 0.0
        if self._inicio is None:
            self._inicio = time.perf_counter()
        self._amostras.append(m)
        self.mensagem = "Calibrando... fique parado!"
        return self.progresso

    def _reiniciar_com(self, mensagem: str) -> None:
        self._amostras = []
        self._inicio = None
        self.mensagem = mensagem

    def resultado(self) -> Calibracao:
        """Mediana das amostras (robusta a quadros ruins)."""
        def med(f) -> float:
            return float(np.median([f(a) for a in self._amostras]))
        nariz_y = float(np.median([a.nariz[1] for a in self._amostras if a.nariz]))
        return Calibracao(
            centro_x=med(lambda a: a.centro_x),
            ombro_y=med(lambda a: a.ombro_y),
            quadril_y=med(lambda a: a.quadril_y),
            nariz_y=nariz_y,
            tronco=med(lambda a: a.tronco),
            largura_ombros=med(lambda a: a.largura_ombros))


# ==========================================================================
# Interpretação: landmarks -> ações e caixas de colisão
# ==========================================================================
def _angulo(a: Ponto, b: Ponto, c: Ponto) -> float:
    """Ângulo em graus no vértice b formado por a-b-c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n < 1e-6:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


class Interpretador:
    """Transforma os landmarks suavizados em `Corpo` (caixas + mãos + ações).

    Todos os limiares são FRAÇÕES do tronco calibrado (L), então funcionam
    para qualquer altura de jogador ou distância da câmera. Cada ação usa
    HISTERESE (limiar para entrar > limiar para sair) para não piscar.

    Regras (y cresce para BAIXO na imagem):
      AGACHAR : (ombro_y - ombro_y_calibrado)/L  >= 0.35   (ombros descem)
                ou ângulo do joelho < 120° com ombros já ≥ metade do limiar
      PULAR   : (quadril_y_calibrado - quadril_y)/L >= 0.08 (quadril sobe);
                o estado "pulando" é mantido por 0.7 s (tempo de voo tolerante)
      INCLINAR: ângulo do eixo quadril->ombro com a vertical >= 15°
      MÃO ALTA: punho acima da linha dos ombros em 0.10 L
    """

    def __init__(self, calibracao: Calibracao) -> None:
        self.cal = calibracao
        self._agachado = False
        self._pulando = False
        self._pulo_ate = 0.0
        self._inclinacao = 0

    def interpretar(self, pontos: List[Ponto]) -> Corpo:
        cal, L = self.cal, self.cal.tronco
        m = medir(pontos)

        # ----- métricas -----
        queda_ombro = (m.ombro_y - cal.ombro_y) / L
        subida_quadril = (cal.quadril_y - m.quadril_y) / L
        angulo_tronco = math.degrees(math.atan2(m.ombro_x - m.quadril_x,
                                                max(m.quadril_y - m.ombro_y, 1.0)))
        angulo_joelho = self._angulo_joelho(pontos)

        # ----- ações com histerese -----
        entra, sai = cfg.AGACHAR_ENTRA_FRAC, cfg.AGACHAR_SAI_FRAC
        joelho_dobrado = (angulo_joelho is not None and angulo_joelho
                          < (cfg.JOELHO_SAI_GRAUS if self._agachado else cfg.JOELHO_AGACHAR_GRAUS)
                          and queda_ombro >= sai * 0.5)
        self._agachado = (queda_ombro >= (sai if self._agachado else entra)) or joelho_dobrado

        # PULAR: ao detectar a subida do quadril, o pulo vale por PULO_DURACAO_S
        # (renovado enquanto o quadril ainda estiver acima do limiar de saída).
        agora = time.perf_counter()
        if subida_quadril >= cfg.PULAR_ENTRA_FRAC:
            self._pulo_ate = agora + cfg.PULO_DURACAO_S
        elif subida_quadril >= cfg.PULAR_SAI_FRAC and self._pulando:
            self._pulo_ate = max(self._pulo_ate, agora + 0.15)
        self._pulando = agora < self._pulo_ate

        limite = cfg.INCLINAR_SAI_GRAUS if self._inclinacao else cfg.INCLINAR_ENTRA_GRAUS
        if abs(angulo_tronco) >= limite:
            self._inclinacao = 1 if angulo_tronco > 0 else -1
        else:
            self._inclinacao = 0

        linha_mao = m.ombro_y - cfg.MAO_LEVANTADA_FRAC * L
        mao_esq = _visivel(pontos, PUNHO_E) and pontos[PUNHO_E][1] < linha_mao
        mao_dir = _visivel(pontos, PUNHO_D) and pontos[PUNHO_D][1] < linha_mao

        acoes = Acoes(self._agachado, self._pulando, self._inclinacao, mao_esq, mao_dir,
                      queda_ombro, subida_quadril, angulo_tronco, angulo_joelho)

        # ----- caixas de colisão (cabeça e tronco) -----
        if m.nariz is not None:
            nx, ny = m.nariz
        else:  # cabeça fora do quadro/invisível: estima pelo tronco
            nx, ny = m.ombro_x, m.ombro_y - cfg.CABECA_NARIZ_ACIMA_OMBRO * L
        meio = cfg.CABECA_LADO_FRAC * L / 2
        cabeca = Retangulo(nx - meio, ny - meio, nx + meio, ny + meio)

        # x do tronco: mistura ombros (mais sensíveis à inclinação) e quadris
        tx = cfg.TRONCO_PESO_OMBRO * m.ombro_x + (1 - cfg.TRONCO_PESO_OMBRO) * m.quadril_x
        meia_larg = cfg.TRONCO_LARGURA_FRAC * L / 2
        tronco = Retangulo(tx - meia_larg, m.ombro_y, tx + meia_larg,
                           m.quadril_y + cfg.TRONCO_EXTENSAO_BAIXO_FRAC * L)

        maos = [(pontos[i][0], pontos[i][1])
                for i in (PUNHO_E, PUNHO_D, INDICE_E, INDICE_D) if _visivel(pontos, i)]
        return Corpo(cabeca, tronco, maos, acoes)

    @staticmethod
    def _angulo_joelho(pontos: List[Ponto]) -> Optional[float]:
        """Média do ângulo quadril-joelho-tornozelo dos lados visíveis."""
        angulos = []
        for q, j, t in ((QUADRIL_E, JOELHO_E, TORNOZELO_E), (QUADRIL_D, JOELHO_D, TORNOZELO_D)):
            if all(_visivel(pontos, i) for i in (q, j, t)):
                angulos.append(_angulo(pontos[q], pontos[j], pontos[t]))
        return sum(angulos) / len(angulos) if angulos else None
