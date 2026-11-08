"""Constantes do jogo "Esquiva com o corpo".

Todos os limiares de detecção, velocidades, cores e caminhos ficam aqui para
que se possa "afinar" o jogo sem tocar na lógica nem na visão computacional.

Convenção: valores chamados *_FRAC são frações do comprimento do tronco (L),
medido na calibração (distância ombro -> quadril, em pixels). Assim as zonas
de agachar/pular se adaptam ao tamanho do jogador e à distância da câmera.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos (sempre relativos à pasta do projeto)
# --------------------------------------------------------------------------
PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_ASSETS = PASTA_PROJETO / "assets"
PASTA_SONS = PASTA_ASSETS / "sons_glam"
CAMINHO_MODELO = PASTA_ASSETS / "pose_landmarker.task"

# Modelo oficial do Google (MediaPipe Model Zoo, "Pose Landmarker lite").
# Fonte: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# --------------------------------------------------------------------------
# Captura
# --------------------------------------------------------------------------
LARGURA = 640
ALTURA = 480
FPS_CAMERA = 30
NOME_JANELA = "Esquiva Glam - Pink Tech (MediaPipe Pose)"

# --------------------------------------------------------------------------
# Visão: confiança, suavização e presença
# --------------------------------------------------------------------------
CONF_DETECCAO = 0.5            # min_pose_detection_confidence
CONF_PRESENCA = 0.5            # min_pose_presence_confidence
CONF_RASTREIO = 0.5            # min_tracking_confidence

LIMIAR_VISIBILIDADE = 0.5      # ombros/quadris abaixo disso => corpo "perdido"
LIMIAR_VIS_OPCIONAL = 0.4      # mãos, joelhos, tornozelos, nariz
MARGEM_BORDA_BAIXO = 0.985     # quadril abaixo dessa fração da altura = fora do quadro

# Suavização exponencial ADAPTATIVA: parado => alfa baixo (remove tremor);
# movendo rápido => alfa alto (evita atraso). alfa = f(deslocamento em px).
ALFA_MIN = 0.35
ALFA_MAX = 0.85
ALFA_DIST_PX = 35.0            # deslocamento a partir do qual usa ALFA_MAX
REINICIO_FILTRO_S = 0.5        # sem detecção por tanto tempo => zera o filtro

# --------------------------------------------------------------------------
# Calibração
# --------------------------------------------------------------------------
CALIBRACAO_SEG = 2.0           # tempo parado e visível para concluir
CALIB_MIN_AMOSTRAS = 10
CALIB_TRONCO_MIN_PX = 60       # tronco menor que isso => jogador longe demais
CALIB_TRONCO_MAX_FRAC = 0.42   # tronco maior que isso*ALTURA => perto demais
CALIB_ESTABILIDADE_FRAC = 0.10 # variação máxima do ombro (em L) durante a calibração

# --------------------------------------------------------------------------
# Regras de detecção das ações (frações de L; histerese entra/sai)
# --------------------------------------------------------------------------
# AGACHAR: os ombros descem em relação ao valor calibrado.
AGACHAR_ENTRA_FRAC = 0.35
AGACHAR_SAI_FRAC = 0.25
# Auxiliar: ângulo do joelho (quadril-joelho-tornozelo) quando as pernas aparecem.
JOELHO_AGACHAR_GRAUS = 120.0
JOELHO_SAI_GRAUS = 135.0
# PULAR: o quadril sobe em relação ao valor calibrado.
PULAR_ENTRA_FRAC = 0.08   # basta um pulinho; o pulo é "travado" por PULO_DURACAO_S
PULAR_SAI_FRAC = 0.04
# Tempo (s) em que o pulo continua valendo depois de detectado. Um pulo real dura
# ~0.4 s e a visão tem atraso, então sem isso seria preciso acertar o instante exato.
PULO_DURACAO_S = 0.7
# INCLINAR: ângulo do eixo quadril->ombro em relação à vertical.
INCLINAR_ENTRA_GRAUS = 15.0
INCLINAR_SAI_GRAUS = 10.0
# MÃO LEVANTADA: punho acima da linha dos ombros.
MAO_LEVANTADA_FRAC = 0.10

# --------------------------------------------------------------------------
# Caixas de colisão do corpo (frações de L)
# --------------------------------------------------------------------------
CABECA_LADO_FRAC = 0.55        # caixa quadrada centrada no nariz
CABECA_NARIZ_ACIMA_OMBRO = 0.45  # usado se o nariz estiver invisível
TRONCO_LARGURA_FRAC = 0.60     # largura total da caixa do tronco
TRONCO_EXTENSAO_BAIXO_FRAC = 0.10  # caixa vai um pouco abaixo do quadril
TRONCO_PESO_OMBRO = 0.70       # centro_x do tronco = 70% ombros + 30% quadril
MAO_RAIO_PX = 16               # tolerância extra para pegar item com o punho

# --------------------------------------------------------------------------
# Obstáculos (geometria relativa ao corpo calibrado)
# --------------------------------------------------------------------------
OBST_LARGURA_PX = 70
ALTO_FOLGA_FRAC = 0.45         # base da barra alta fica 0.45L acima dos ombros
BAIXO_TOPO_FRAC = 0.02         # topo do bloco baixo fica 0.02L acima do quadril
LATERAL_BORDA_FRAC = 0.20      # borda interna da pedra, a partir do centro (em L)
LATERAL_LARGURA_FRAC = 0.90
LATERAL_ALTURA_FRAC = 1.30
LATERAL_AVISO_S = 0.9          # tempo de alerta antes da pedra cair
RASPAO_FRAC = 0.05             # folga < 0.05L ao passar => esquiva "raspando"

# --------------------------------------------------------------------------
# Regras do jogo
# --------------------------------------------------------------------------
VIDAS_INICIAIS = 3
ESCUDOS_MAX = 2
INVULNERAVEL_DANO_S = 1.5
INVULNERAVEL_ESCUDO_S = 0.8
TOLERANCIA_PERDA_S = 0.5       # sem pessoa por esse tempo => pausa automática
REGRESSO_ESTAVEL_S = 0.3       # pessoa precisa ficar visível para retomar
CONTAGEM_S = 3

PONTOS_POR_SEGUNDO = 5
PONTOS_ESQUIVA = 100
PONTOS_RASPAO = 150
PONTOS_MOEDA = 50
PONTOS_ESCUDO = 30
BONUS_FASE = 200
COMBO_POR_NIVEL = 4            # a cada 4 acertos seguidos sobe o multiplicador
MULT_MAX = 5

# Dificuldade progressiva por fase
FASE_DURACAO_S = 15.0
VEL_BASE = 230.0               # px/s
VEL_PASSO = 45.0
VEL_MAX = 520.0
INTERVALO_BASE_S = 2.3         # tempo entre obstáculos
INTERVALO_PASSO_S = 0.2
INTERVALO_MIN_S = 0.9
GAP_BASE_S = 1.1               # separação mínima entre chegadas ao jogador
GAP_PASSO_S = 0.05
GAP_MIN_S = 0.7
FASE_LATERAL = 2               # a partir de qual fase surgem pedras laterais
ITEM_INTERVALO_S = (3.5, 6.0)
ITEM_RAIO_PX = 18
ITEM_CHANCE_ESCUDO = 0.2

# --------------------------------------------------------------------------
# Cores (BGR, padrão do OpenCV)
# --------------------------------------------------------------------------
# Tema "pink tech" (inspirado no universo glam/rosa; paleta própria)
COR_BRANCO = (255, 255, 255)
COR_PRETO = (0, 0, 0)
COR_ROSA = (180, 105, 255)          # rosa choque
COR_PINK_FORTE = (138, 33, 224)     # pink intenso
COR_ROSA_CLARO = (230, 192, 255)
COR_MAGENTA = (200, 40, 255)
COR_LILAS = (255, 150, 200)
COR_AMEIXA = (95, 20, 85)           # contorno escuro dos textos
COR_DOURADO = (90, 215, 255)
COR_CIANO = (255, 235, 40)          # ciano neon (acento "tech")
COR_VERDE = (200, 255, 130)         # menta (confirmações)
COR_VERMELHO = (110, 60, 255)       # dano
COR_AMARELO = COR_DOURADO
COR_LARANJA = (120, 150, 255)       # raspão
COR_AZUL = (255, 200, 120)
COR_CINZA = (215, 190, 235)
COR_ESQUELETO = (255, 230, 250)
COR_CABECA = COR_CIANO
COR_TRONCO = (200, 160, 255)

# Sons e trilha (gerados com numpy/wave na primeira execução; só no Windows)
SOM_VOLUME = 0.40
MUSICA_VOLUME = 280                # 0..1000 (MCI)
BPM_MUSICA = 118
