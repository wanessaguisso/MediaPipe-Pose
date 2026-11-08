# Execução:
#   pip install -r requirements.txt
#   python main.py                      (webcam 0)
#   python main.py --camera 1           (outra câmera)
#   python main.py --video caminho.mp4  (arquivo de vídeo)
"""Ponto de entrada: argumentos de linha de comando e loop principal.

Cada quadro: câmera -> PoseLandmarker -> Interpretador (ações) -> Jogo (regras)
-> visual (desenho) -> janela. A lógica nunca vê imagens, só `Entrada`.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np

import config as cfg
from logica import Estado, Jogo
from modelos import Entrada
from pose import (Calibrador, ErroFonte, ErroModelo, FonteVideo, Interpretador,
                  RastreadorPose, garantir_modelo)
from som import Som
from visual import Efeitos, desenhar_cena

TECLAS_SAIR = (ord("q"), ord("Q"), 27)


def ler_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Esquiva com o corpo (MediaPipe Pose)")
    p.add_argument("--camera", type=int, default=0, help="índice da webcam (padrão: 0)")
    p.add_argument("--video", type=str, default=None,
                   help="usa um arquivo de vídeo em vez da webcam")
    p.add_argument("--sem-repetir", action="store_true",
                   help="encerra quando o vídeo acabar (padrão: reinicia)")
    p.add_argument("--sem-som", action="store_true", help="desliga os efeitos sonoros")
    p.add_argument("--debug", action="store_true", help="inicia com o modo debug ligado")
    p.add_argument("--janela", action="store_true",
                   help="inicia em janela (padrão: tela cheia; tecla F alterna)")
    return p.parse_args()


def definir_tela_cheia(ativa: bool) -> None:
    cv2.setWindowProperty(cfg.NOME_JANELA, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if ativa else cv2.WINDOW_NORMAL)
    if not ativa:
        cv2.resizeWindow(cfg.NOME_JANELA, cfg.LARGURA, cfg.ALTURA)


def exibir(quadro, tela_cheia: bool) -> None:
    """Mostra o quadro; em tela cheia amplia mantendo a proporção (barras pretas)."""
    if tela_cheia:
        _, _, w, h = cv2.getWindowImageRect(cfg.NOME_JANELA)
        if w > 0 and h > 0:
            escala = min(w / quadro.shape[1], h / quadro.shape[0])
            nw, nh = int(quadro.shape[1] * escala), int(quadro.shape[0] * escala)
            grande = cv2.resize(quadro, (nw, nh), interpolation=cv2.INTER_LINEAR)
            tela = np.zeros((h, w, 3), np.uint8)
            x0, y0 = (w - nw) // 2, (h - nh) // 2
            tela[y0:y0 + nh, x0:x0 + nw] = grande
            quadro = tela
    cv2.imshow(cfg.NOME_JANELA, quadro)


def tratar_tecla(tecla: int, jogo: Jogo, calibrador: Calibrador, debug: bool) -> Optional[bool]:
    """Aplica o comando do teclado. Devolve o novo valor de `debug`, ou None para sair."""
    if tecla in TECLAS_SAIR:
        return None
    if tecla == 32:                               # ESPAÇO
        if jogo.estado == Estado.MENU:
            calibrador.resetar()
        jogo.alternar_espaco()
    elif tecla in (ord("r"), ord("R")):
        if jogo.calibracao is None:
            calibrador.resetar()
        jogo.reiniciar()
    elif tecla in (ord("c"), ord("C")):
        calibrador.resetar()
        jogo.recalibrar()
    elif tecla in (ord("d"), ord("D")):
        debug = not debug
    elif tecla in (ord("n"), ord("N")) and debug:  # atalho de depuração p/ demonstração
        jogo.avancar_fase()
    return debug


def main() -> int:
    args = ler_argumentos()
    try:
        caminho_modelo = garantir_modelo()
        fonte = FonteVideo(args.camera, args.video, repetir=not args.sem_repetir)
        rastreador = RastreadorPose(caminho_modelo)
    except (ErroModelo, ErroFonte) as erro:
        print(f"[erro] {erro}", file=sys.stderr)
        return 1

    jogo = Jogo()
    calibrador = Calibrador()
    interpretador: Optional[Interpretador] = None
    efeitos = Efeitos()
    som = Som(ativo=not args.sem_som)
    som.iniciar_musica()
    debug = args.debug
    fps = 0.0
    anterior = time.perf_counter()

    cv2.namedWindow(cfg.NOME_JANELA, cv2.WINDOW_NORMAL)
    tela_cheia = not args.janela
    definir_tela_cheia(tela_cheia)
    print("[jogo] ESPACO inicia | R reinicia | C recalibra | D debug | F tela cheia | Q/ESC sai")
    try:
        while True:
            quadro = fonte.ler()
            if quadro is None:
                print("[jogo] Fim do vídeo ou a câmera parou de responder.")
                break

            # ---- visão ----
            leitura = rastreador.processar(quadro)

            # ---- calibração (só enquanto o jogo está calibrando) ----
            progresso, mensagem = 0.0, ""
            if jogo.estado == Estado.CALIBRANDO:
                progresso = calibrador.alimentar(leitura)
                mensagem = calibrador.mensagem
                if calibrador.pronto:
                    jogo.definir_calibracao(calibrador.resultado())
                    interpretador = Interpretador(jogo.calibracao)

            corpo = None
            if jogo.calibracao is not None and interpretador is not None \
                    and leitura.presente and leitura.pontos is not None:
                corpo = interpretador.interpretar(leitura.pontos)

            # ---- regras do jogo ----
            agora = time.perf_counter()
            dt = min(agora - anterior, 0.1)   # limita saltos (ex.: janela arrastada)
            anterior = agora
            fps = 1.0 / dt if fps == 0.0 else 0.9 * fps + 0.1 / max(dt, 1e-3)
            jogo.atualizar(dt, Entrada(leitura.presente, corpo, progresso, mensagem))

            eventos = jogo.coletar_eventos()
            efeitos.processar(eventos)
            efeitos.atualizar(dt)
            som.reagir(eventos)
            som.manter_musica()

            # ---- desenho ----
            desenhar_cena(quadro, jogo, leitura, corpo, efeitos, fps, debug)
            exibir(quadro, tela_cheia)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("f"), ord("F")):
                tela_cheia = not tela_cheia
                definir_tela_cheia(tela_cheia)
            elif tecla != 255:
                novo_debug = tratar_tecla(tecla, jogo, calibrador, debug)
                if novo_debug is None:
                    break
                debug = novo_debug
            if cv2.getWindowProperty(cfg.NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break   # janela fechada no X
    except KeyboardInterrupt:
        pass
    finally:
        som.parar_musica()
        rastreador.fechar()
        fonte.liberar()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
