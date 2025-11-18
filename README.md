# 🔧 AutoTunning

Projeto acadêmico voltado à implementação e estudo de técnicas de autoajuste de parâmetros em algoritmos de otimização.

## 👥 Integrantes
- Victor G. Cavalcante  
- João Vitor Lopes

## Objetivo
Este programa otimiza automaticamente os parâmetros de um modelo `.exe` com 10 parâmetros de entrada (1 textual e 9 numéricos) usando diferentes estratégias de otimização.

## Estratégias Implementadas
- Pattern Search
- Simplex (Nelder-Mead)
- Estratégia combinada: Algoritmo Genético + Particle Swarm

## Configuração
Editar `config.py` para definir:
- `MODE`: 'max' ou 'min'
- `EXE_PATH`: caminho para o `.exe`
- `TIME_LIMIT_SECONDS`: limite de tempo da execução

## Execução
```bash
python AutoTunning.py