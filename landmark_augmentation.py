"""
landmark_augmentation.py
========================
Data augmentation diretamente nos landmarks extraídos pelo MediaPipe.

Por que no landmark e não na imagem?
- Não reprocessa o vídeo com o MediaPipe (muito mais rápido)
- Não depende de bibliotecas externas (só numpy)
- Controle exato sobre o que muda geometricamente
- Funciona igual para modo RF (2D) e LSTM (3D/sequências)

Cada função recebe um frame (lista/array de 63 valores: x1,y1,z1,...,x21,y21,z21)
e retorna uma versão augmentada. Para sequências LSTM, aplica a mesma
transformação em todos os frames da sequência (mantém consistência temporal).

Transformações implementadas:
  1. Ruído gaussiano  — simula tremor de mão / imprecisão do sensor
  2. Escala           — simula distância diferente da câmera
  3. Rotação 2D       — simula ângulo diferente de filmagem (plano XY)
  4. Espelhamento     — simula mão esquerda vs. direita
  5. Composição       — combina ruído + escala + rotação aleatoriamente
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _reshape(frame):
    """Converte lista plana (63,) em matriz (21, 3) para operar por eixo."""
    return np.array(frame, dtype=np.float32).reshape(21, 3)


def _flatten(pts):
    """Converte matriz (21, 3) de volta para lista plana (63,)."""
    return pts.flatten().tolist()


# ---------------------------------------------------------------------------
# Transformações individuais (operam em um único frame)
# ---------------------------------------------------------------------------

def _ruido(pts, intensidade=0.005):
    """
    Adiciona ruído gaussiano pequeno a cada coordenada.
    Simula tremor natural da mão e imprecisão do sensor.
    intensidade: desvio padrão relativo às coordenadas normalizadas.
    """
    ruido = np.random.normal(0, intensidade, pts.shape).astype(np.float32)
    return pts + ruido


def _escala(pts, fator_min=0.85, fator_max=1.15):
    """
    Escala uniformemente todos os landmarks em torno da origem (pulso = 0,0,0).
    Simula a mão mais perto ou mais longe da câmera.
    Como já normalizamos pelo pulso, escalar em torno da origem é correto.
    """
    fator = np.random.uniform(fator_min, fator_max)
    return pts * fator


def _rotacao_2d(pts, angulo_max_graus=15.0):
    """
    Rotaciona os landmarks no plano XY em torno da origem.
    Simula rotação do pulso ou ângulo diferente de filmagem.
    Não rotaciona Z para não distorcer a profundidade artificialmente.
    """
    angulo = np.radians(np.random.uniform(-angulo_max_graus, angulo_max_graus))
    cos_a, sin_a = np.cos(angulo), np.sin(angulo)

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    pts[:, 0] = cos_a * x - sin_a * y
    pts[:, 1] = sin_a * x + cos_a * y
    return pts


def _espelhamento(pts):
    """
    Inverte o eixo X de todos os landmarks.
    Simula a mão esquerda (espelho da mão direita).
    É a augmentation mais poderosa: cria amostras genuinamente diferentes.
    Nota: a inversão de X é válida porque os landmarks já estão normalizados
    pelo pulso — não há translação residual que distorça o espelhamento.
    """
    pts[:, 0] = -pts[:, 0]
    return pts


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def augmentar_frame(frame, transformacoes):
    """
    Aplica uma lista de transformações em sequência a um único frame.

    Parâmetros
    ----------
    frame : list ou np.array de shape (63,)
    transformacoes : list de callables, ex: [_ruido, _escala]

    Retorna
    -------
    list de floats com shape (63,)
    """
    pts = _reshape(frame)
    for t in transformacoes:
        pts = t(pts)
    return _flatten(pts)


def augmentar_sequencia(sequencia, transformacoes):
    """
    Aplica as mesmas transformações em TODOS os frames de uma sequência LSTM.
    É crucial usar a mesma transformação em todos os frames para preservar
    a coerência temporal — se rotacionarmos frame a frame com ângulos
    diferentes, o modelo aprende movimentos que nunca existiram.

    Parâmetros
    ----------
    sequencia : list de frames, shape (T, 63)
    transformacoes : list de callables

    Retorna
    -------
    list de frames augmentados, shape (T, 63)
    """
    return [augmentar_frame(frame, transformacoes) for frame in sequencia]


def gerar_amostras_aumentadas(features, labels, mode="rf", n_aumentos=5, seed=42):
    """
    Gera amostras augmentadas para todo o dataset.

    Para cada amostra original, cria `n_aumentos` variações usando
    combinações aleatórias das transformações disponíveis.

    Parâmetros
    ----------
    features : list
        - mode="rf"  : list de frames,     shape (N, 63)
        - mode="lstm": list de sequências, shape (N, T, 63)
    labels : list de str, shape (N,)
    mode : "rf" ou "lstm"
    n_aumentos : int
        Número de amostras geradas POR amostra original.
        Com 3 vídeos e ~30 frames/vídeo:
          - n_aumentos=5  → 6× mais dados
          - n_aumentos=10 → 11× mais dados
    seed : int
        Semente para reprodutibilidade.

    Retorna
    -------
    features_aug : list (inclui originais + augmentadas)
    labels_aug   : list (inclui originais + augmentadas)
    """
    np.random.seed(seed)

    # Transforma disponíveis (sem espelhamento — tratado separadamente)
    transformacoes_disponiveis = [_ruido, _escala, _rotacao_2d]

    features_aug = list(features)   # começa com os originais
    labels_aug   = list(labels)

    originais_por_classe = {}
    for f, l in zip(features, labels):
        originais_por_classe.setdefault(l, 0)
        originais_por_classe[l] += 1

    print("\n--- Augmentation de Landmarks ---")
    print(f"Amostras originais: {len(features)}")
    print(f"Gerando {n_aumentos} variações por amostra...")

    for i, (amostra, label) in enumerate(zip(features, labels)):
        # 1. Espelhamento (sempre gerado — é a mais valiosa)
        if mode == "rf":
            pts = _reshape(amostra)
            esp = _flatten(_espelhamento(pts.copy()))
            features_aug.append(esp)
        else:
            esp = augmentar_sequencia(amostra, [_espelhamento])
            features_aug.append(esp)
        labels_aug.append(label)

        # 2. Combinações aleatórias das demais transformações
        for _ in range(n_aumentos - 1):
            # Escolhe aleatoriamente quantas e quais transformações aplicar
            n_t = np.random.randint(1, len(transformacoes_disponiveis) + 1)
            escolhidas = np.random.choice(
                transformacoes_disponiveis, size=n_t, replace=False
            ).tolist()

            if mode == "rf":
                nova = augmentar_frame(amostra, escolhidas)
                features_aug.append(nova)
            else:
                nova = augmentar_sequencia(amostra, escolhidas)
                features_aug.append(nova)

            labels_aug.append(label)

    print(f"Amostras após augmentation: {len(features_aug)}")
    print(f"Aumento: {len(features_aug) / max(len(features), 1):.1f}×")

    # Mostra distribuição por classe
    dist = {}
    for l in labels_aug:
        dist[l] = dist.get(l, 0) + 1
    print("\nDistribuição por classe:")
    for classe, qtd in sorted(dist.items()):
        orig = originais_por_classe.get(classe, 0)
        print(f"  {classe}: {orig} → {qtd} amostras")

    return features_aug, labels_aug


# ---------------------------------------------------------------------------
# Teste rápido (rode: python landmark_augmentation.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Teste do módulo de augmentation ===\n")

    # Simula 6 amostras RF (2 gestos × 3 amostras)
    np.random.seed(0)
    features_rf = [np.random.randn(63).tolist() for _ in range(6)]
    labels_rf   = ["oi"] * 3 + ["tchau"] * 3

    f_aug, l_aug = gerar_amostras_aumentadas(
        features_rf, labels_rf, mode="rf", n_aumentos=5
    )
    assert len(f_aug) == len(l_aug), "Tamanhos inconsistentes!"
    assert len(f_aug[0]) == 63, "Shape do frame incorreto!"
    print("\n[RF] OK")

    # Simula 4 sequências LSTM (2 gestos × 2 sequências, T=20)
    features_lstm = [np.random.randn(20, 63).tolist() for _ in range(4)]
    labels_lstm   = ["oi"] * 2 + ["tchau"] * 2

    f_aug2, l_aug2 = gerar_amostras_aumentadas(
        features_lstm, labels_lstm, mode="lstm", n_aumentos=5
    )
    assert len(f_aug2) == len(l_aug2), "Tamanhos inconsistentes!"
    assert len(f_aug2[0]) == 20, "Número de frames incorreto!"
    assert len(f_aug2[0][0]) == 63, "Shape do frame LSTM incorreto!"
    print("[LSTM] OK")
    print("\nTodos os testes passaram!")