"""Som do jogo (tema glam/pink tech), gerado 100% com numpy + wave.

Nada de arquivos de terceiros: os efeitos (sinos, harpa, glitter) e a trilha pop
em loop são sintetizados aqui, com composição original (progressão de acordes
C - Am - F - G, arpejos e batida), criados na primeira execução em
assets/sons_glam/.

Reprodução (somente Windows, bibliotecas padrão):
  - efeitos: `winsound.PlaySound` (assíncrono)
  - trilha : MCI (`winmm.mciSendStringW`), que toca em paralelo aos efeitos
Em outros sistemas o módulo fica mudo.
"""
from __future__ import annotations

import ctypes
import sys
import time
import wave
from typing import Dict, List, Tuple

import numpy as np

import config as cfg
from logica import Evento, TipoEvento

TAXA = 22050
ALIAS_MCI = "glam_bgm"

# Frequências (Hz) das notas usadas (afinação padrão, A4 = 440)
N = {"C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.00,
     "B4": 493.88, "C5": 523.25, "D5": 587.33, "E5": 659.25, "F5": 698.46, "G5": 783.99,
     "A5": 880.00, "B5": 987.77, "C6": 1046.50, "D6": 1174.66, "E6": 1318.51,
     "G6": 1567.98, "C7": 2093.00, "E7": 2637.02, "G3": 196.00, "A3": 220.00,
     "F3": 174.61, "C3": 130.81, "A2": 110.00, "F2": 87.31, "G2": 98.00, "C2": 65.41}


# ---------------------------------------------------------------------------
# Síntese
# ---------------------------------------------------------------------------
def _sino(freq: float, dur: float, vol: float = 1.0) -> np.ndarray:
    """Sino/glockenspiel: parciais inarmônicos com decaimento exponencial."""
    t = np.arange(int(TAXA * dur)) / TAXA
    onda = (np.sin(2 * np.pi * freq * t)
            + 0.45 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t * 14)
            + 0.25 * np.sin(2 * np.pi * freq * 5.40 * t) * np.exp(-t * 25))
    ataque = np.minimum(1.0, t / 0.004)
    return vol * onda * np.exp(-t * (5.0 / max(dur, 0.1))) * ataque


def _pluck(freq: float, dur: float, vol: float = 1.0) -> np.ndarray:
    """Nota "pluck" de synth: seno + harmônicos com decaimento rápido."""
    t = np.arange(int(TAXA * dur)) / TAXA
    onda = (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t) * np.exp(-t * 12)
            + 0.25 * np.sin(6 * np.pi * freq * t) * np.exp(-t * 20))
    return vol * onda * np.exp(-t * 9) * np.minimum(1.0, t / 0.003)


def _baixo(freq: float, dur: float, vol: float = 1.0) -> np.ndarray:
    t = np.arange(int(TAXA * dur)) / TAXA
    onda = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    return vol * onda * np.exp(-t * 4) * np.minimum(1.0, t / 0.005)


def _ruido(dur: float, vol: float, brilho: float = 1.0, queda: float = 40.0,
           rng=np.random.default_rng(7)) -> np.ndarray:
    """Ruído curto (hi-hat/palma). brilho=1 deixa agudo (diferença do ruído)."""
    n = int(TAXA * dur)
    r = rng.uniform(-1, 1, n)
    r = np.diff(r, prepend=0.0) * brilho + r * (1 - brilho) * 0.4
    t = np.arange(n) / TAXA
    return vol * r * np.exp(-t * queda)


def _colocar(buf: np.ndarray, sinal: np.ndarray, inicio_s: float) -> None:
    i = int(inicio_s * TAXA)
    fim = min(len(buf), i + len(sinal))
    if i < len(buf):
        buf[i:fim] += sinal[:fim - i]


def _arpejo(notas: List[str], passo: float, dur: float, vol: float = 0.6,
            sino: bool = True) -> np.ndarray:
    total = passo * len(notas) + dur
    buf = np.zeros(int(TAXA * total))
    for k, nome in enumerate(notas):
        gerar = _sino if sino else _pluck
        _colocar(buf, gerar(N[nome], dur, vol), k * passo)
    return buf


def _glissando(inicio: str, fim: str, passos: int, passo: float, dur: float = 0.35,
               vol: float = 0.5) -> np.ndarray:
    """"Glitter": harpa subindo rápido (escala pentatônica entre duas notas)."""
    escala = [N[k] for k in ("C5", "D5", "E5", "G5", "A5", "C6", "D6", "E6", "G6", "C7", "E7")]
    f0, f1 = N[inicio], N[fim]
    freqs = [f for f in escala if f0 <= f <= f1] or [f0, f1]
    freqs = [freqs[int(i * (len(freqs) - 1) / max(passos - 1, 1))] for i in range(passos)]
    total = passo * passos + dur
    buf = np.zeros(int(TAXA * total))
    for k, f in enumerate(freqs):
        _colocar(buf, _sino(f, dur, vol), k * passo)
    return buf


def _slide_triste(f0: float, f1: float, dur: float, vol: float = 0.6) -> np.ndarray:
    """Tom que desce com leve vibrato ("ops!" cartunesco, suave)."""
    t = np.arange(int(TAXA * dur)) / TAXA
    freq = f0 + (f1 - f0) * (t / dur)
    fase = 2 * np.pi * np.cumsum(freq + 8 * np.sin(2 * np.pi * 6 * t)) / TAXA
    return vol * np.sin(fase) * np.exp(-t * 3) * np.minimum(1.0, t / 0.01)


def _efeitos() -> Dict[str, np.ndarray]:
    return {
        "moeda": _arpejo(["E6", "C7"], 0.07, 0.35, 0.7),
        "escudo": _arpejo(["C6", "E6", "G6", "C7"], 0.06, 0.4, 0.6),
        "escudo_usado": _arpejo(["G5", "E5", "C5"], 0.07, 0.3, 0.6),
        "esquiva": _arpejo(["G5", "C6"], 0.05, 0.22, 0.45),
        "raspao": _glissando("C5", "E7", 9, 0.035, 0.35, 0.5),
        "dano": _slide_triste(520, 190, 0.45, 0.7),
        "fase": np.concatenate([_arpejo(["C5", "E5", "G5", "C6", "E6", "G6"], 0.07, 0.35, 0.55),
                                _glissando("C6", "E7", 8, 0.04, 0.4, 0.45)]),
        "gameover": _arpejo(["E5", "D5", "C5", "A4"], 0.18, 0.7, 0.6),
        "tick": _sino(N["G5"], 0.18, 0.6),
        "inicio": _glissando("C5", "E7", 11, 0.04, 0.45, 0.6),
    }


def _musica() -> np.ndarray:
    """Trilha pop de 8 compassos em loop: baixo, arpejo, batida e sinos."""
    beat = 60.0 / cfg.BPM_MUSICA
    compasso = 4 * beat
    cauda = 1.5
    buf = np.zeros(int(TAXA * (8 * compasso + cauda)))
    # (acorde, baixo, notas do arpejo) para C - Am - F - G
    acordes = [("C", "C2", ["C4", "E4", "G4", "C5", "G4", "E4", "G4", "E4"]),
               ("Am", "A2", ["A3", "C4", "E4", "A4", "E4", "C4", "E4", "C4"]),
               ("F", "F2", ["F3", "A3", "C4", "F4", "C4", "A3", "C4", "A3"]),
               ("G", "G2", ["G3", "D4", "G4", "B4", "G4", "D4", "G4", "D4"])]
    for c in range(8):
        _, baixo, arp = acordes[c % 4]
        t0 = c * compasso
        for k in range(8):                              # arpejo em colcheias
            _colocar(buf, _pluck(N[arp[k]], beat, 0.35), t0 + k * beat / 2)
        for b in (0, 2):                                # baixo nos tempos 1 e 3
            _colocar(buf, _baixo(N[baixo], beat * 1.5, 0.9), t0 + b * beat)
        for b in range(4):                              # bumbo "soft" + hi-hat
            _colocar(buf, _baixo(55.0, 0.12, 0.8) * np.exp(-np.arange(int(TAXA * 0.12)) / TAXA * 20),
                     t0 + b * beat)
            _colocar(buf, _ruido(0.05, 0.25, 1.0, 70), t0 + b * beat + beat / 2)
        for b in (1, 3):                                # "palma" nos tempos 2 e 4
            _colocar(buf, _ruido(0.12, 0.45, 0.6, 28), t0 + b * beat)
        melodia = {0: "E6", 1: "D6", 2: "C6", 3: "G5"}[c % 4]
        _colocar(buf, _sino(N[melodia], beat * 2, 0.35), t0 + 2 * beat)   # sino no meio
        if c % 4 == 0:
            _colocar(buf, _glissando("C6", "E7", 6, 0.05, 0.3, 0.25), t0)
    # loop sem emenda: a cauda que passou do fim volta a somar no começo
    n = int(8 * compasso * TAXA)
    buf[:len(buf) - n] += buf[n:]
    return buf[:n]


def _salvar(caminho, audio: np.ndarray, volume: float, pico: float = 0.9) -> None:
    audio = audio / max(np.max(np.abs(audio)), 1e-6) * pico * min(1.0, volume)
    fade = min(len(audio) // 20, int(0.01 * TAXA))
    if fade:
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)
    with wave.open(str(caminho), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TAXA)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())


EVENTO_PARA_SOM = {
    TipoEvento.DANO: "dano",
    TipoEvento.ESCUDO_USADO: "escudo_usado",
    TipoEvento.MOEDA: "moeda",
    TipoEvento.ESCUDO: "escudo",
    TipoEvento.ESQUIVA: "esquiva",
    TipoEvento.RASPAO: "raspao",
    TipoEvento.FASE: "fase",
    TipoEvento.GAME_OVER: "gameover",
    TipoEvento.CONTAGEM: "tick",
    TipoEvento.INICIO: "inicio",
}


# ---------------------------------------------------------------------------
# Reprodução
# ---------------------------------------------------------------------------
class Som:
    def __init__(self, ativo: bool = True) -> None:
        self.ativo = ativo and sys.platform == "win32"
        self._winsound = None
        self._mci = None
        self._musica_ligada = False
        self._ultima_checagem = 0.0
        if not self.ativo:
            return
        try:
            import winsound
            cfg.PASTA_SONS.mkdir(parents=True, exist_ok=True)
            for nome, audio in _efeitos().items():
                caminho = cfg.PASTA_SONS / f"{nome}.wav"
                if not caminho.exists():
                    _salvar(caminho, audio, cfg.SOM_VOLUME * 2)
            if not (cfg.PASTA_SONS / "trilha.wav").exists():
                _salvar(cfg.PASTA_SONS / "trilha.wav", _musica(), 0.6)
            self._winsound = winsound
            self._mci = ctypes.windll.winmm
        except (ImportError, OSError, AttributeError) as erro:
            print(f"[som] Desativado: {erro}")
            self.ativo = False

    # --- efeitos ---
    def tocar(self, nome: str) -> None:
        if not self.ativo:
            return
        try:
            self._winsound.PlaySound(str(cfg.PASTA_SONS / f"{nome}.wav"),
                                     self._winsound.SND_FILENAME | self._winsound.SND_ASYNC)
        except RuntimeError:
            pass

    def reagir(self, eventos: List[Evento]) -> None:
        """Toca o som do evento mais importante do quadro (o último da lista)."""
        for ev in reversed(eventos):
            nome = EVENTO_PARA_SOM.get(ev.tipo)
            if nome:
                self.tocar(nome)
                return

    # --- trilha (MCI) ---
    def _mci_cmd(self, comando: str) -> str:
        buf = ctypes.create_unicode_buffer(128)
        self._mci.mciSendStringW(comando, buf, 127, None)
        return buf.value

    def iniciar_musica(self) -> None:
        if not self.ativo:
            return
        caminho = cfg.PASTA_SONS / "trilha.wav"
        self._mci_cmd(f'open "{caminho}" type waveaudio alias {ALIAS_MCI}')
        self._mci_cmd(f"setaudio {ALIAS_MCI} volume to {cfg.MUSICA_VOLUME}")
        self._mci_cmd(f"play {ALIAS_MCI} from 0")
        self._musica_ligada = True

    def manter_musica(self) -> None:
        """Chamar a cada quadro: se a trilha acabou, recomeça (loop)."""
        if not self._musica_ligada:
            return
        agora = time.perf_counter()
        if agora - self._ultima_checagem < 0.5:
            return
        self._ultima_checagem = agora
        if self._mci_cmd(f"status {ALIAS_MCI} mode") == "stopped":
            self._mci_cmd(f"play {ALIAS_MCI} from 0")

    def parar_musica(self) -> None:
        if self._musica_ligada:
            self._mci_cmd(f"stop {ALIAS_MCI}")
            self._mci_cmd(f"close {ALIAS_MCI}")
            self._musica_ligada = False
