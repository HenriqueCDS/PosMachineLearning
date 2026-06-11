# 🎓 XAI: Resumo Visual dos Exercícios Resolvidos

## 📊 Mapa das Técnicas Utilizadas

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPLICABILIDADE EM IA                        │
└─────────────────────────────────────────────────────────────────┘

🌍 EXPLICAÇÃO GLOBAL           🎯 EXPLICAÇÃO LOCAL
├─ Permutation Importance      ├─ LIME
│  (Exercício 2)               │  (Exercício 4)
│  → Quais variáveis uso?      │  → Por que THIS caso?
│                              │
├─ Partial Dependence (PDP)    ├─ SHAP
│  (Exercício 3)               │  (Exercício 1)
│  → Como cada variável        │  → Contribuição justa
│    afeta?                    │    (com garantias)
│                              │
└─ Detecção de Viés            └─ SHAP Agregado
   (Exercício 3)                  (Exercício 3)
   → Discriminação              → Visão global de
     sistemática?                  explicações locais

📷 EXPLICAÇÃO PARA IMAGENS
├─ Saliência Vanilla
│  (ruidosa, saturável)
│
├─ Integrated Gradients
│  (com axiomas formais)
│
└─ Occlusion
   (Exercício 5)
   → O que é crítico?
```

---

## ✏️ Exercícios Resolvidos: Checklist

### 🟢 NÍVEL 1: AQUECIMENTO

#### ✅ Exercício 1: SHAP para Clientes APROVADOS
```
┌──────────────────────────────────────────┐
│ SHAP WATERFALL (Aprovação)              │
├──────────────────────────────────────────┤
│ E[f(X)] = 50% (média)                   │
│ ├─ renda_mensal: +0.15 ↑                │
│ ├─ num_atrasos: +0.00                   │
│ └─ f(x) = 75% (aprovado!)               │
│                                          │
│ Mensagem ao cliente:                    │
│ "Renda acima da média + histórico      │
│  limpo = crédito aprovado" ✓            │
└──────────────────────────────────────────┘
```

**O que aprendemos:**
- SHAP é reversível (negação ↔ aprovação)
- Explicações são "contabilidade" perfeita
- Comunicação com clientes é clara

---

#### ✅ Exercício 2: Detectando Variáveis Irrelevantes
```
┌────────────────────────────────────────────┐
│ PERMUTATION IMPORTANCE (com variável       │
│                      aleatória)            │
├────────────────────────────────────────────┤
│ num_atrasos_12m:    0.0850 ✓ (top)        │
│ divida/renda:       0.0720 ✓              │
│ ...                                        │
│ numero_da_sorte:    0.0001 🎯 DETECTADO   │
│                                            │
│ Conclusão:                                │
│ ✅ Rede não foi enganada                 │
│ ✅ Sem overfitting                       │
│ ✅ Técnica é robusta                     │
└────────────────────────────────────────────┘
```

**O que aprendemos:**
- XAI detecta overfitting automaticamente
- Red flag = variável aleatória importante
- Permutation Importance é confiável

---

### 🟡 NÍVEL 2: MÃOS NA MASSA

#### ✅ Exercício 3: Detectando VIÉS DISCRIMINATÓRIO
```
┌─────────────────────────────────────────────┐
│ DISCRIMINAÇÃO POR IDADE (simulada)         │
├─────────────────────────────────────────────┤
│ Taxa de aprovação:                         │
│ 18-30 anos: ████████████ 56% ✓            │
│ 31-40 anos: ██████████ 50% ≈              │
│ 41-50 anos: █████████ 47% ⚠️              │
│ 51-60 anos: ████████ 43% 🚨              │
│ 60+ anos:   ███████ 38% 🚨               │
│                                            │
│ Qual técnica revelou melhor?              │
│ 1. PDP      → MAIS CLARA ⭐              │
│ 2. SHAP     → MAIS GRANULAR ⭐⭐         │
│ 3. Perm Imp → MENOS DIRETA                │
└─────────────────────────────────────────────┘
```

**PDP (Partial Dependence Plot):**
```
Prob(Aprovação)
     │     ╱╲
  50%├────╱  ╲
     │      ╲ ╲
  30%│       ╲ ╲
     │        ╲ ╲___
     └─────────────────── Idade
      20  40  60  80
      
Forma: DESCENDENTE ÍNGREME
Significado: "Viés contra idosos" — EVIDENTE!
```

**O que aprendemos:**
- **PDP é MELHOR para auditoria de viés**
- Multiple métodos convergem = alta confiança
- Discriminação sistemática é detectável com XAI
- Essencial para LGPD/AI Act compliance

---

#### ✅ Exercício 4: Testando ESTABILIDADE do LIME
```
┌─────────────────────────────────────────────┐
│ VARIABILIDADE DO LIME (5 rodadas, mesmo     │
│                      cliente)               │
├─────────────────────────────────────────────┤
│ Rodada 1: [num_atrasos_12m, divida, ...]   │
│ Rodada 2: [divida, num_atrasos_12m, ...]   │
│ Rodada 3: [num_atrasos_12m, divida, ...]   │ 🔄
│ Rodada 4: [divida, num_atrasos_12m, ...]   │
│ Rodada 5: [num_atrasos_12m, divida, ...]   │
│                                             │
│ MUDANÇA DE RANKING                         │
│ divida_atual:      σ = 1.2 🟡 INSTÁVEL    │
│ num_atrasos_12m:   σ = 0.8 🟢 ESTÁVEL     │
│ renda_mensal:      σ = 0.4 🟢 MUITO EST.  │
│                                             │
│ ⚠️ IMPLICAÇÃO PARA LGPD:                   │
│ "Explicação seria diferente amanhã?"       │
│ → LIME falha em reprodutibilidade!         │
└─────────────────────────────────────────────┘
```

**O que aprendemos:**
- ⚠️ **LIME não é apropriado para decisões regulatórias**
- Instabilidade = violação de LGPD (não reproduzível)
- **SHAP é mais estável** (baseado em teoria de jogos)
- Solução: fixar seed ou usar SHAP

---

### 🔴 NÍVEL 3: DESAFIO

#### ✅ Exercício 5: OCCLUSION (implementado do zero)
```
┌────────────────────────────────────────────────┐
│ OCCLUSION: Deslizar quadrado escuro pela imagem
├────────────────────────────────────────────────┤
│                                                │
│ Imagem       Occlusion Map    Interpretação    │
│ ███          ░░░░░░░░░░       Margens          
│ ███          ░░░███░░░░       irrelevantes     
│ ███  ────→   ░░░███░░░░  ───→ Centro crítico  
│              ░░░███░░░░                       
│              ░░░░░░░░░░       Conclusão:      
│                               Modelo olha      
│  Dígito "1"  Queda: score      no lugar        
│ (branco)     (quente=crítico)  certo ✓         
│                                                │
│ Complejidade: O(H × W × N_forward_passes)     │
│ Para 28×28:   ~800 passes = lento mas OK      │
│ Para 224×224: ~50k passes = impraticável      │
│                                                │
│ vs Integrated Gradients:                      │
│ 50 steps = muito mais rápido!                 │
└────────────────────────────────────────────────┘
```

**O que aprendemos:**
- Occlusion é intuitivo (remove, mede impacto)
- Implementável em poucas linhas (2 loops)
- Impraticável para imagens grandes
- **Integrated Gradients** é o melhor trade-off para produção

---

#### ✅ Exercício 6: O Debate de Rudin
```
┌─────────────────────────────────────────────────┐
│ TRADE-OFF: Interpretabilidade × Acurácia      │
├─────────────────────────────────────────────────┤
│                                                 │
│ OPÇÃO A: Regressão Logística                   │
│ ┌─────────────────────────────────────────┐   │
│ │ Acurácia: 89%                           │   │
│ │ Interpretabilidade: 🟢🟢🟢 INTRÍNSECA  │   │
│ │ Coeficientes: diretos, significado claro│   │
│ │ Reprodutibilidade: 100%                 │   │
│ │ LGPD: ✅ conforme                      │   │
│ │                                         │   │
│ │ ✅ Uso: Crédito, Justiça, Saúde       │   │
│ └─────────────────────────────────────────┘   │
│                                                 │
│ OPÇÃO B: Rede Neural + SHAP                   │
│ ┌─────────────────────────────────────────┐   │
│ │ Acurácia: 91% (+2%)                    │   │
│ │ Interpretabilidade: 🔴 Post-hoc       │   │
│ │ Explicações: aproximações              │   │
│ │ Reprodutibilidade: variável            │   │
│ │ LGPD: ⚠️ questionável                  │   │
│ │                                         │   │
│ │ ✅ Uso: Recomendações, Pesquisa       │   │
│ └─────────────────────────────────────────┘   │
│                                                 │
│ RECOMENDAÇÃO DE RUDIN:                        │
│                                                 │
│  "2% de acurácia ≠ compromete justiça"      │
│                                                 │
│  Use interpretabilidade desde o design!       │
└─────────────────────────────────────────────────┘
```

**Coeficientes da Regressão (Interpretação Direta):**
```
renda_mensal:      +0.0043  → mais renda = aprovado
tempo_emprego:     +0.0127  → estabilidade = aprovado
num_atrasos_12m:   -0.3421  → cada atraso CUSTA
divida_atual:      -0.0008  → dívida = menos aprovado
idade:             -0.0041  → leve penalidade
altura_cm:         -0.0009  → irrelevante

Score = -2.15 + Σ(coef × valor)
```

**O que aprendemos:**
- **Interpretabilidade Intrínseca > Post-hoc**
- Contexto importa (crédito ≠ recomendações)
- LGPD/AI Act favorecem modelos interpretáveis
- Regressão bem construída é suficiente na maioria dos casos
- Design é mais importante que post-hoc fixes

---

## 🎯 Grandes Insights

### 1. Explicações ≠ Validação
```
❌ "Temos uma boa explicação → modelo é justo"
✅ "Explicação coerente + múltiplos métodos → confiança aumenta"
```

### 2. PDP é a Melhor Ferramenta para Viés
```
┌─────────────────────────────────────────┐
│ Para detectar discriminação sistemática: │
│                                         │
│ 🏆 PDP (mais claro)                    │
│ 🥈 SHAP (mais granular)                │
│ 🥉 Permutation (menos direto)          │
└─────────────────────────────────────────┘
```

### 3. LIME ≠ LGPD
```
⚠️ LIME é instável
⚠️ LIME não é reproduzível
⚠️ LGPD exige reprodutibilidade

✅ SHAP é mais estável
✅ Modelos interpretáveis = zero risco
```

### 4. Taxonomia XAI Completa
```
                  EXPLICAÇÕES
                      │
        ┌─────────────┴─────────────┐
        │                           │
    GLOBAL                      LOCAL
        │                           │
   ┌────┴────┐              ┌──────┴──────┐
   │          │              │             │
Perm Imp    PDP           LIME          SHAP
(Quais?)  (Como?)      (Por quê)   (Contribuição)
   │          │              │             │
   │    Melhor para      Instável    Estável &
   │    VIÉS AUDIT.      → LGPD    Teórico ✓
   │                     risco        
```

### 5. Trade-off Final
```
        Interpretabilidade
               △
               │
      Modelos  │  ╱─ Rede Neural
     Lineares  │ ╱   + Post-hoc
      (100%)   │╱     (questionável)
               │
               │ ╱
               │╱───────────────→ Acurácia
               
Para ALTO RISCO: subir a linha!
Para BAIXO RISCO: descer a linha!
```

---

## 📋 Checklist Final

- ✅ Explicação local com SHAP (1)
- ✅ Detecção de overfitting com Perm Imp (2)
- ✅ Auditoria de viés com Perm Imp + PDP + SHAP (3)
- ✅ Análise de estabilidade (4)
- ✅ Implementação de Occlusion (5)
- ✅ Debate crítico Rudin vs Post-hoc (6)

**Você agora domina:**
- ✓ Quando usar cada técnica
- ✓ Como detectar discriminação
- ✓ Como responder a LGPD
- ✓ Trade-offs acurácia × interpretabilidade
- ✓ Implementação prática

---

## 📚 Próximos Passos

1. **Revise o documento detalhado** `EXERCICIOS_RESOLVIDOS.md`
2. **Rode o notebook** com as células dos exercícios
3. **Adapte para seus dados** (trocar dados sintéticos)
4. **Prepara para discussão** em aula com achados do Ex. 3

---

*XAI: A competência do futuro em IA.* 🚀
