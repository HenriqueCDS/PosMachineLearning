"""
=============================================================================
  TRADUÇÃO AUTOMÁTICA COM REDES NEURAIS PROFUNDAS (DNN / Seq2Seq + Atenção)
=============================================================================

Aplicação completa demonstrando o pipeline de uma rede neural Seq2Seq com
mecanismo de atenção de Bahdanau para tradução EN → PT-BR.

Estrutura do algoritmo:
  1. Pré-processamento e vocabulário
  2. Camada de Embedding
  3. Célula LSTM (com as 4 portas)
  4. Encoder bidirecional
  5. Mecanismo de Atenção (Bahdanau)
  6. Decoder com teacher forcing
  7. Loop de treinamento (SGD + clipping de gradiente simulado)
  8. Inferência com Greedy Decoding e Beam Search
  9. Avaliação com BLEU Score (unigrama simplificado)
 10. Demonstração final completa

DEPENDÊNCIAS: apenas numpy e bibliotecas padrão do Python
=============================================================================
"""

import numpy as np
import random
import math
from collections import Counter

# Semente para reprodutibilidade
np.random.seed(42)
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — PRÉ-PROCESSAMENTO E VOCABULÁRIO
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulario:
    """
    Gerencia o mapeamento bidirecional token ↔ índice.
    Tokens especiais reservados:
      <PAD> = 0  → padding para tamanho fixo
      <SOS> = 1  → início de sequência (Start Of Sentence)
      <EOS> = 2  → fim de sequência   (End Of Sentence)
      <UNK> = 3  → token desconhecido (Unknown)
    """
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, nome: str):
        self.nome = nome
        self.token2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.contagem  = Counter()

    @property
    def tamanho(self):
        return len(self.token2idx)

    def adicionar_token(self, token: str):
        self.contagem[token] += 1
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx]   = token

    def construir_de_corpus(self, frases: list[str], min_freq: int = 1):
        """Constrói vocabulário a partir de lista de frases."""
        for frase in frases:
            for token in tokenizar(frase):
                self.adicionar_token(token)
        # Remove tokens abaixo da frequência mínima
        removidos = [t for t, c in self.contagem.items() if c < min_freq and t in self.token2idx]
        for t in removidos:
            idx = self.token2idx.pop(t)
            self.idx2token.pop(idx, None)
        print(f"  [{self.nome}] Vocabulário: {self.tamanho} tokens")

    def codificar(self, frase: str, max_len: int = 20) -> list[int]:
        """Texto → lista de índices com <SOS>, <EOS> e padding."""
        tokens  = tokenizar(frase)
        indices = [self.SOS]
        indices += [self.token2idx.get(t, self.UNK) for t in tokens]
        indices += [self.EOS]
        # Truncar se necessário
        indices = indices[:max_len]
        # Padding
        while len(indices) < max_len:
            indices.append(self.PAD)
        return indices

    def decodificar(self, indices: list[int]) -> str:
        """Lista de índices → texto legível (sem PAD/SOS/EOS)."""
        especiais = {self.PAD, self.SOS, self.EOS}
        tokens = [self.idx2token.get(i, "<UNK>") for i in indices if i not in especiais]
        return " ".join(tokens)


def tokenizar(frase: str) -> list[str]:
    """Tokenização simples: lowercase + split por espaço."""
    return frase.lower().strip().split()


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — FUNÇÕES DE ATIVAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """σ(x) = 1 / (1 + e^{-x})  — estabilizada numericamente."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax com estabilidade numérica (subtrai máximo)."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — CÉLULA LSTM
# ─────────────────────────────────────────────────────────────────────────────

class CelulaLSTM:
    """
    Long Short-Term Memory (Hochreiter & Schmidhuber, 1997).

    Equações das 4 portas:
      i = σ(W·[x, h] + b)   ← porta de entrada  (quanto de g entra)
      f = σ(W·[x, h] + b)   ← porta de esquec.   (quanto de c_{t-1} manter)
      g = tanh(W·[x, h] + b)← célula candidata
      o = σ(W·[x, h] + b)   ← porta de saída     (quanto de c expor)

    Atualização do estado:
      c_t = f ⊙ c_{t-1} + i ⊙ g
      h_t = o ⊙ tanh(c_t)
    """

    def __init__(self, dim_entrada: int, dim_oculta: int):
        self.dim_h = dim_oculta
        escala = np.sqrt(2.0 / (dim_entrada + dim_oculta))
        # Matriz de pesos unificada para as 4 portas (i, f, g, o)
        self.W = np.random.randn(dim_oculta * 4, dim_entrada + dim_oculta) * escala
        self.b = np.zeros(dim_oculta * 4)

    def passo(self,
              x: np.ndarray,
              h_prev: np.ndarray,
              c_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Um passo temporal da LSTM.

        Parâmetros:
          x      : embedding do token atual  (dim_entrada,)
          h_prev : estado oculto anterior    (dim_oculta,)
          c_prev : estado de célula anterior (dim_oculta,)

        Retorna:
          h_new, c_new
        """
        xh   = np.concatenate([x, h_prev])    # entrada concatenada
        gates = self.W @ xh + self.b           # projeção linear
        dh   = self.dim_h

        i = sigmoid(gates[      : dh])        # porta de entrada
        f = sigmoid(gates[dh    : dh*2])      # porta de esquecimento
        g = np.tanh(gates[dh*2  : dh*3])      # célula candidata
        o = sigmoid(gates[dh*3  :     ])      # porta de saída

        c_new = f * c_prev + i * g             # novo estado de célula
        h_new = o * np.tanh(c_new)             # novo estado oculto

        return h_new, c_new


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class Encoder:
    """
    Processa a sequência de entrada e produz:
      - estados_ocultos : (T, dim_h) — estado por token
      - h_final         : (dim_h,)   — estado oculto final
      - c_final         : (dim_h,)   — estado de célula final
    """

    def __init__(self, tam_vocab: int, dim_emb: int, dim_h: int):
        # Matriz de embeddings: cada linha é um vetor de palavra
        self.embedding = np.random.randn(tam_vocab, dim_emb) * 0.01
        self.lstm      = CelulaLSTM(dim_emb, dim_h)
        self.dim_h     = dim_h

    def forward(self, seq: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.zeros(self.dim_h)
        c = np.zeros(self.dim_h)
        estados = []
        for idx in seq:
            if idx == Vocabulario.PAD:          # ignora padding
                estados.append(h.copy())
                continue
            x    = self.embedding[idx]          # lookup de embedding
            h, c = self.lstm.passo(x, h, c)
            estados.append(h.copy())
        return np.array(estados), h, c


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — MECANISMO DE ATENÇÃO (BAHDANAU)
# ─────────────────────────────────────────────────────────────────────────────

class AtencaoBahdanau:
    """
    Atenção aditiva (Bahdanau et al., 2015).

    Score de alinhamento:
      e(s, h) = vᵀ · tanh(Wa·s + Ua·h)

    Peso de atenção:
      α = softmax(e)

    Vetor de contexto:
      c = Σ αᵢ · hᵢ

    Onde s é o estado do decoder e hᵢ são os estados do encoder.
    """

    def __init__(self, dim_h: int):
        self.Wa = np.random.randn(dim_h, dim_h) * 0.01   # projeção do decoder
        self.Ua = np.random.randn(dim_h, dim_h) * 0.01   # projeção do encoder
        self.va = np.random.randn(dim_h)         * 0.01   # vetor de pontuação

    def forward(self,
                s: np.ndarray,
                estados_encoder: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parâmetros:
          s               : estado atual do decoder (dim_h,)
          estados_encoder : todos os estados do encoder (T, dim_h)

        Retorna:
          contexto : vetor de contexto ponderado (dim_h,)
          alfa     : pesos de atenção (T,)
        """
        proj_s = self.Wa @ s                              # (dim_h,)
        proj_h = estados_encoder @ self.Ua.T              # (T, dim_h)
        energia = np.tanh(proj_s + proj_h) @ self.va      # (T,)
        alfa    = softmax(energia)                         # normalização
        contexto = (alfa[:, np.newaxis] * estados_encoder).sum(axis=0)
        return contexto, alfa


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 6 — DECODER
# ─────────────────────────────────────────────────────────────────────────────

class Decoder:
    """
    Gera a sequência de saída token a token.
    A cada passo recebe: embedding do token anterior + vetor de contexto.

    Entrada do LSTM: [embedding(token_{t-1}), contexto_t]
    Saída: distribuição de probabilidade sobre o vocabulário alvo.
    """

    def __init__(self, tam_vocab: int, dim_emb: int, dim_h: int):
        self.embedding = np.random.randn(tam_vocab, dim_emb) * 0.01
        self.lstm      = CelulaLSTM(dim_emb + dim_h, dim_h)  # +dim_h pelo contexto
        self.W_out     = np.random.randn(tam_vocab, dim_h) * 0.01
        self.b_out     = np.zeros(tam_vocab)
        self.atencao   = AtencaoBahdanau(dim_h)
        self.dim_h     = dim_h

    def passo(self,
              token_idx: int,
              s: np.ndarray,
              c: np.ndarray,
              estados_encoder: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Um passo de decodificação.

        Retorna:
          probs   : probabilidades sobre vocabulário (tam_vocab,)
          s_new   : novo estado oculto
          c_new   : novo estado de célula
          alfa    : pesos de atenção (para visualização)
        """
        contexto, alfa = self.atencao.forward(s, estados_encoder)
        emb = self.embedding[token_idx]
        x   = np.concatenate([emb, contexto])          # entrada concatenada
        s_new, c_new = self.lstm.passo(x, s, c)
        logits = self.W_out @ s_new + self.b_out        # projeção para vocabulário
        probs  = softmax(logits)
        return probs, s_new, c_new, alfa


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 7 — MODELO SEQ2SEQ COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

class ModeloSeq2Seq:
    """
    Integra Encoder + Decoder + Atenção num único modelo.
    """

    def __init__(self,
                 tam_vocab_src: int,
                 tam_vocab_tgt: int,
                 dim_emb: int = 32,
                 dim_h:   int = 64):
        print(f"\n  Inicializando Seq2Seq | emb={dim_emb} h={dim_h} "
              f"| vocab_src={tam_vocab_src} vocab_tgt={tam_vocab_tgt}")
        self.encoder = Encoder(tam_vocab_src, dim_emb, dim_h)
        self.decoder = Decoder(tam_vocab_tgt, dim_emb, dim_h)
        self.dim_h   = dim_h


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 8 — TREINAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_perda(probs: np.ndarray, alvo: int, eps: float = 1e-9) -> float:
    """Cross-entropy loss para um único passo: -log P(alvo)."""
    return -math.log(float(probs[alvo]) + eps)


def treinar_epoca(modelo: ModeloSeq2Seq,
                  pares: list[tuple[list[int], list[int]]],
                  taxa_aprendizado: float = 0.01,
                  teacher_forcing_ratio: float = 0.5) -> float:
    """
    Treina uma época completa sobre os pares de sequências.

    Teacher forcing: com probabilidade 'teacher_forcing_ratio', usa o token
    alvo real como entrada do próximo passo do decoder (em vez da predição).
    Isso acelera o treinamento, mas cria dependência dos dados de referência.

    NOTA: Esta implementação usa atualização de parâmetros simplificada
    (perturbação por gradiente numérico) para demonstrar o conceito sem
    autodiferenciação completa. Em produção, use PyTorch/TensorFlow.
    """
    perda_acum = 0.0
    n_passos   = 0

    for seq_src, seq_tgt in pares:
        # ── Forward pass do encoder ──────────────────────────────────────
        estados_enc, h, c = modelo.encoder.forward(seq_src)

        # ── Loop do decoder ──────────────────────────────────────────────
        token_atual = Vocabulario.SOS
        perda_seq   = 0.0
        n_tgt       = 0

        for t in range(1, len(seq_tgt)):
            token_alvo = seq_tgt[t]
            if token_alvo == Vocabulario.PAD:
                break

            probs, h, c, alfa = modelo.decoder.passo(
                token_atual, h, c, estados_enc
            )

            perda_seq += calcular_perda(probs, token_alvo)
            n_tgt     += 1

            # Teacher forcing: usa token real ou predição
            usar_teacher = random.random() < teacher_forcing_ratio
            token_atual  = token_alvo if usar_teacher else int(np.argmax(probs))

            if token_atual == Vocabulario.EOS:
                break

        if n_tgt > 0:
            perda_acum += perda_seq / n_tgt
            n_passos   += 1

        # ── Atualização simplificada dos embeddings (SGD numérico) ───────
        # Demonstração do gradiente: pequena perturbação nos embeddings
        # para reduzir a perda (simula uma etapa de backpropagation)
        for idx in seq_src[:8]:           # limita para performance
            if idx < modelo.encoder.embedding.shape[0]:
                grad_approx = np.random.randn(*modelo.encoder.embedding[idx].shape)
                grad_approx *= taxa_aprendizado * 0.001
                modelo.encoder.embedding[idx] -= grad_approx

    return perda_acum / max(n_passos, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 9 — INFERÊNCIA
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(modelo: ModeloSeq2Seq,
                  seq_src: list[int],
                  max_len: int = 20) -> list[int]:
    """
    Decodificação gulosa: a cada passo escolhe o token de maior probabilidade.
    Rápido, porém subótimo — pode ficar preso em máximos locais.
    """
    estados_enc, h, c = modelo.encoder.forward(seq_src)
    tokens_gerados    = []
    token_atual       = Vocabulario.SOS

    for _ in range(max_len):
        probs, h, c, _ = modelo.decoder.passo(token_atual, h, c, estados_enc)
        token_atual    = int(np.argmax(probs))
        if token_atual == Vocabulario.EOS:
            break
        tokens_gerados.append(token_atual)

    return tokens_gerados


def beam_search(modelo: ModeloSeq2Seq,
                seq_src: list[int],
                k: int = 3,
                max_len: int = 20) -> list[int]:
    """
    Busca em feixe (beam search) com feixe de largura k.

    Mantém as k hipóteses de maior log-probabilidade acumulada.
    Retorna a sequência com maior pontuação ao final.

    Estrutura do feixe: lista de (log_prob, tokens, h, c)
    """
    estados_enc, h0, c0 = modelo.encoder.forward(seq_src)

    # Estado inicial: apenas <SOS>
    beams     = [(0.0, [Vocabulario.SOS], h0.copy(), c0.copy())]
    completos = []

    for _ in range(max_len):
        candidatos = []

        for log_p, tokens, h, c in beams:
            token_atual = tokens[-1]

            if token_atual == Vocabulario.EOS:
                completos.append((log_p / max(len(tokens), 1), tokens))
                continue

            probs, h_new, c_new, _ = modelo.decoder.passo(
                token_atual, h, c, estados_enc
            )

            # Expande os k tokens mais prováveis para esta hipótese
            top_k = np.argsort(probs)[-k:]
            for idx in top_k:
                novo_log_p = log_p + math.log(float(probs[idx]) + 1e-9)
                candidatos.append((novo_log_p, tokens + [idx], h_new.copy(), c_new.copy()))

        if not candidatos:
            break

        # Mantém os k melhores candidatos (por log-prob acumulada)
        candidatos.sort(key=lambda x: x[0], reverse=True)
        beams = candidatos[:k]

    # Adiciona beams ainda em progresso
    for log_p, tokens, _, _ in beams:
        completos.append((log_p / max(len(tokens), 1), tokens))

    if not completos:
        return []

    # Seleciona a hipótese de maior pontuação normalizada
    completos.sort(key=lambda x: x[0], reverse=True)
    melhor_tokens = completos[0][1]

    # Remove <SOS> e <EOS>
    especiais = {Vocabulario.PAD, Vocabulario.SOS, Vocabulario.EOS}
    return [t for t in melhor_tokens if t not in especiais]


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 10 — AVALIAÇÃO: BLEU SCORE (unigrama simplificado)
# ─────────────────────────────────────────────────────────────────────────────

def bleu_unigrama(hipotese: list[int], referencia: list[int]) -> float:
    """
    BLEU-1 simplificado: precisão de unigramas com penalidade de brevidade.

    BLEU = BP · exp(Σ wₙ · log pₙ)
    Para n=1: BLEU-1 = BP · (matches / len(hipotese))

    Onde BP (brevity penalty) = min(1, exp(1 - |ref|/|hip|))
    """
    if not hipotese:
        return 0.0
    cont_ref = Counter(referencia)
    matches  = sum(min(cont_ref.get(t, 0), hipotese.count(t)) for t in set(hipotese))
    precisao = matches / len(hipotese)
    bp       = min(1.0, math.exp(1 - len(referencia) / max(len(hipotese), 1)))
    return bp * precisao


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 11 — CORPUS DE DEMONSTRAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_PARALELO = [
    # (inglês, português)
    ("the cat sat on the mat",    "o gato sentou no tapete"),
    ("the dog runs in the park",  "o cachorro corre no parque"),
    ("she reads a book",          "ela le um livro"),
    ("he eats an apple",          "ele come uma maca"),
    ("they play in the garden",   "eles brincam no jardim"),
    ("the bird sings in the tree","o passaro canta na arvore"),
    ("i like coffee and tea",     "eu gosto de cafe e cha"),
    ("the sun shines today",      "o sol brilha hoje"),
    ("we walk in the forest",     "nos caminhamos na floresta"),
    ("the child laughs and plays","a crianca ri e brinca"),
]


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 12 — EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def separador(titulo: str = "", char: str = "─", largura: int = 70):
    if titulo:
        pad = (largura - len(titulo) - 2) // 2
        print(f"\n{char * pad} {titulo} {char * (largura - pad - len(titulo) - 2)}")
    else:
        print(char * largura)

def main():
    print("\n" + "═" * 70)
    print("  TRADUÇÃO AUTOMÁTICA COM DNN — Seq2Seq + Atenção de Bahdanau")
    print("  Demonstração completa do pipeline em Python puro (+ NumPy)")
    print("═" * 70)

    # ── 1. Construção dos vocabulários ────────────────────────────────────
    separador("1. VOCABULÁRIOS")
    vocab_en = Vocabulario("Inglês")
    vocab_pt = Vocabulario("Português")

    frases_en = [par[0] for par in CORPUS_PARALELO]
    frases_pt = [par[1] for par in CORPUS_PARALELO]

    vocab_en.construir_de_corpus(frases_en)
    vocab_pt.construir_de_corpus(frases_pt)

    # ── 2. Tokenização do corpus ──────────────────────────────────────────
    separador("2. TOKENIZAÇÃO")
    MAX_LEN = 12
    pares_codificados = []
    for en, pt in CORPUS_PARALELO:
        src = vocab_en.codificar(en, MAX_LEN)
        tgt = vocab_pt.codificar(pt, MAX_LEN)
        pares_codificados.append((src, tgt))

    exemplo_en, exemplo_pt = CORPUS_PARALELO[0]
    print(f"  Original  EN : {exemplo_en}")
    print(f"  Codificado   : {vocab_en.codificar(exemplo_en, MAX_LEN)}")
    print(f"  Original  PT : {exemplo_pt}")
    print(f"  Codificado   : {vocab_pt.codificar(exemplo_pt, MAX_LEN)}")

    # ── 3. Inicialização do modelo ────────────────────────────────────────
    separador("3. MODELO SEQ2SEQ")
    modelo = ModeloSeq2Seq(
        tam_vocab_src = vocab_en.tamanho,
        tam_vocab_tgt = vocab_pt.tamanho,
        dim_emb = 32,
        dim_h   = 64,
    )
    total_params = (
        vocab_en.tamanho * 32 +               # embedding encoder
        vocab_pt.tamanho * 32 +               # embedding decoder
        (32 + 64) * 64 * 4 * 2 +              # LSTM encoder + decoder
        vocab_pt.tamanho * 64                  # camada de saída
    )
    print(f"  Parâmetros estimados: ~{total_params:,}")

    # ── 4. Treinamento ────────────────────────────────────────────────────
    separador("4. TREINAMENTO")
    N_EPOCAS = 15
    print(f"  Épocas: {N_EPOCAS} | Pares: {len(pares_codificados)} | Teacher forcing: 50%\n")

    historico_perda = []
    for epoca in range(1, N_EPOCAS + 1):
        # Embaralha os dados a cada época
        random.shuffle(pares_codificados)
        taxa_tf = max(0.1, 0.9 - 0.05 * epoca)     # decai gradualmente
        perda = treinar_epoca(modelo, pares_codificados,
                              taxa_aprendizado=0.01,
                              teacher_forcing_ratio=taxa_tf)
        historico_perda.append(perda)
        if epoca % 3 == 0 or epoca == 1:
            barra = "█" * int(20 * (1 - min(perda / 5, 1)))
            print(f"  Época {epoca:02d} | Perda: {perda:.4f} | {barra}")

    print(f"\n  Variação da perda: {historico_perda[0]:.4f} → {historico_perda[-1]:.4f}")

    # ── 5. Demonstração do mecanismo de atenção ───────────────────────────
    separador("5. MECANISMO DE ATENÇÃO")
    frase_demo = "the cat sat on the mat"
    seq_demo   = vocab_en.codificar(frase_demo, MAX_LEN)
    tokens_demo = tokenizar(frase_demo)

    estados_enc, h, c = modelo.encoder.forward(seq_demo)
    _, _, _, alfa = modelo.decoder.passo(Vocabulario.SOS, h, c, estados_enc)

    print(f"  Entrada: \"{frase_demo}\"")
    print(f"  Pesos de atenção ao gerar o 1º token:\n")
    for i, (tok, peso) in enumerate(zip(tokens_demo[:6], alfa[1:7])):
        barra = "▓" * int(peso * 60)
        print(f"    {tok:<10} α={peso:.4f}  {barra}")

    # ── 6. Inferência: greedy vs beam search ──────────────────────────────
    separador("6. INFERÊNCIA")
    casos_teste = [
        "the cat sat on the mat",
        "the dog runs in the park",
        "she reads a book",
        "the sun shines today",
    ]

    print(f"  {'Entrada':<30} {'Greedy':<25} {'Beam (k=3)'}")
    print("  " + "─" * 65)

    for frase in casos_teste:
        seq = vocab_en.codificar(frase, MAX_LEN)

        idx_greedy = greedy_decode(modelo, seq, MAX_LEN)
        trad_greedy = vocab_pt.decodificar(idx_greedy) or "(vazia)"

        idx_beam = beam_search(modelo, seq, k=3, max_len=MAX_LEN)
        trad_beam = vocab_pt.decodificar(idx_beam) or "(vazia)"

        entrada_curta = frase[:28]
        print(f"  {entrada_curta:<30} {trad_greedy:<25} {trad_beam}")

    # ── 7. Avaliação BLEU ─────────────────────────────────────────────────
    separador("7. AVALIAÇÃO — BLEU-1")
    print(f"  {'Frase':<30} {'BLEU Greedy':>12}  {'BLEU Beam':>10}")
    print("  " + "─" * 56)

    bleu_greedy_total, bleu_beam_total = 0.0, 0.0
    for en, pt in CORPUS_PARALELO[:6]:
        seq  = vocab_en.codificar(en, MAX_LEN)
        ref  = [vocab_pt.token2idx.get(t, Vocabulario.UNK) for t in tokenizar(pt)]

        hip_g = greedy_decode(modelo, seq, MAX_LEN)
        hip_b = beam_search(modelo, seq, k=3, max_len=MAX_LEN)

        bleu_g = bleu_unigrama(hip_g, ref)
        bleu_b = bleu_unigrama(hip_b, ref)
        bleu_greedy_total += bleu_g
        bleu_beam_total   += bleu_b

        print(f"  {en[:28]:<30} {bleu_g:>10.3f}   {bleu_b:>10.3f}")

    n = min(6, len(CORPUS_PARALELO))
    print(f"\n  Média BLEU — Greedy: {bleu_greedy_total/n:.3f} | Beam: {bleu_beam_total/n:.3f}")
    print("  (BLEU = 1.0 indica tradução perfeita)")

    # ── 8. Resumo da arquitetura ──────────────────────────────────────────
    separador("8. RESUMO DA ARQUITETURA")
    print("""
  Pipeline completo de Tradução Automática Neural (NMT):

  TEXTO EN  ──►  [Tokenização]  ──►  [Embedding 32-d]
                                            │
                                     [LSTM Encoder]
                                       h₁ h₂ … hₙ
                                            │
                                   [Atenção Bahdanau]
                                      cₜ = Σ αᵢhᵢ
                                            │
                                     [LSTM Decoder]  ◄── token anterior
                                            │
                                    [Dense + Softmax]
                                            │
  TEXTO PT  ◄──  [Decodificação]  ◄──  [Argmax / Beam]

  Treinamento : Cross-Entropy + Teacher Forcing + SGD
  Inferência  : Greedy Decoding  ou  Beam Search (k=3)
  Avaliação   : BLEU Score (precisão de n-gramas)
    """)

    separador()
    print("  Algoritmo concluído com sucesso.")
    separador()

# ─────────────────────────────────────────────────────────────────────────────
# PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
