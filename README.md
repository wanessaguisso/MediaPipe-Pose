# Esquiva Glam – pink tech (MediaPipe Pose)

> Tema visual e sonoro "pink tech glam": gradiente rosa com grade, painéis de vidro, glitter animado,
> esqueleto neon, obstáculos brilhantes (barra pink, caixa glam, cristal) e gemas/moedas.
> Tudo é desenhado com OpenCV; a trilha pop (com sinos, harpa e glitter) e os efeitos são **sintetizados
> com numpy** (composição original, nenhum áudio ou imagem de terceiros).

Projeto de Visão Computacional (FAESA 2026/2 – Conceito 2). Obstáculos atravessam a tela e o
jogador desvia **com o corpo** na frente da webcam. Usa a API **MediaPipe Tasks** (`PoseLandmarker`,
modo VIDEO) e uma lógica de jogo própria construída sobre os landmarks.

## Instalação

```bash
pip install -r requirements.txt
```

O modelo `assets/pose_landmarker.task` (~6 MB) é baixado automaticamente na primeira execução
(precisa de internet só nessa vez). Se o download falhar, o programa mostra o link para baixar à mão.

## Execução

```bash
python main.py                       # webcam 0
python main.py --camera 1            # outra câmera
python main.py --video caminho.mp4   # arquivo de vídeo (reinicia ao acabar; use --sem-repetir para encerrar)
python main.py --sem-som --debug     # sem som / já com o modo debug ligado
```

## Controles

| Tecla | Ação |
|---|---|
| `ESPAÇO` | iniciar (vai para a calibração) / pausar / retomar / jogar de novo |
| `R` | reiniciar a partida (mantém a calibração) |
| `C` | recalibrar |
| `F` | alterna tela cheia / janela (o jogo já inicia em tela cheia; `--janela` inicia em janela) |
| `D` | modo debug (landmarks, visibilidade, valores dos limiares) |
| `N` | (só no modo debug) pula para a próxima fase – útil para demonstração |
| `Q` / `ESC` | sair |

## Como jogar

1. Aperte `ESPAÇO` e fique **em pé, de frente**, mostrando da cabeça ao quadril. Em ~2 s o jogo mede o
   seu tronco; todas as zonas passam a ser **relativas ao seu corpo**, não a pixels fixos.
2. Contagem 3-2-1 e começa:
   - **Barra alta (vermelha)** → **agache**.
   - **Bloco baixo (laranja)** → **pule**.
   - **Pedra lateral** (coluna vermelha piscando avisa onde vai cair) → **incline o tronco** para o lado oposto.
   - **Moedas e escudos** → encoste o **punho** neles.
3. Bater com a **cabeça ou o tronco** custa uma vida (o escudo absorve uma batida). Zero vidas = game over.
4. Saiu do quadro? O jogo **pausa sozinho** ("Volte para a frente da camera") e retoma com contagem quando você volta.

## Mecânicas próprias

- **Três tipos de obstáculo**, cada um exigindo uma ação diferente (agachar / pular / inclinar).
- **Itens coletáveis** com a mão: moeda (+pontos) e escudo (absorve dano, máximo 2).
- **Pontuação** por tempo sobrevivido, por esquiva, bônus por esquiva **"raspando"** (folga < 5% do tronco),
  **combo** com multiplicador até x5 (zera ao levar dano).
- **Fases** a cada 15 s: velocidade e frequência crescentes; pedras laterais aparecem a partir da fase 2.
- Gerador que **espaça as chegadas** dos obstáculos, para nunca exigir duas ações simultâneas.
- **Calibração** que define zonas de agachar/pular, altura das barras e posição das pedras.

## Como a visão vira regra de jogo (`pose.py`)

Todos os limiares são frações do tronco calibrado (L = distância ombro → quadril), com histerese:

| Ação | Regra |
|---|---|
| Agachado | ombros descem ≥ 0,35 L (auxílio: ângulo do joelho < 120°) |
| Pulando | quadril sobe ≥ 0,16 L |
| Inclinado | eixo quadril→ombro ≥ 15° da vertical |
| Mão alta | punho 0,10 L acima dos ombros |

Os landmarks passam por média móvel exponencial adaptativa (parado = suave, movendo = rápido) e pontos com
`visibility`/`presence` abaixo de 0,5 são ignorados. Se ombros/quadris ficarem invisíveis ou saírem do quadro,
o jogador é considerado ausente.

## Estrutura

```
main.py       ponto de entrada, argumentos, loop principal e teclas
pose.py       câmera/vídeo, download do modelo, PoseLandmarker, suavização, calibração, ações
logica.py     regras: máquina de estados, obstáculos, colisões, pontos, combo, fases (sem cv2/mediapipe)
visual.py     desenho com OpenCV: esqueleto, zonas, HUD, efeitos, telas
modelos.py    tipos compartilhados (Retangulo, Corpo, Acoes, Calibracao, Entrada)
som.py        trilha em loop + efeitos gerados com numpy/wave (Windows; mudo em outros sistemas)
config.py     todas as constantes (limiares, velocidades, cores, caminhos)
assets/       modelo baixado e sons gerados em assets/sons_glam (criados na 1ª execução)
requirements.txt
```

## Dicas e observações

- Deixe o corpo inteiro do tronco e os quadris no quadro, boa luz e câmera na altura do peito, a ~2 m.
- Se mexer a câmera ou mudar de lugar, aperte `C` para recalibrar.
- O OpenCV não desenha acentos, por isso os textos da tela aparecem sem acentuação.
- Origem do modelo: Google MediaPipe Model Zoo (`pose_landmarker_lite`). Nenhum outro asset de terceiros é usado.
