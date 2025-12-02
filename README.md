# 🚀 Auto Tuning de Executáveis

Este repositório contém uma ferramenta robusta em Python desenvolvida para realizar a **otimização automática de parâmetros** de softwares externos (executáveis "caixa-preta").

O script utiliza meta-heurísticas avançadas para encontrar a melhor combinação de **10 parâmetros inteiros** (no intervalo de 1 a 1000) que maximizam ou minimizam a saída do seu programa.

## 📋 Funcionalidades

* **Otimização Black-Box:** Não requer acesso ao código-fonte do executável alvo.
* **Multiestratégia:** Executa e compara três abordagens automaticamente:
    1.  🧬 **Algoritmo Genético (GA):** Evolução baseada em seleção, crossover e mutação.
    2.  🐦 **Particle Swarm Optimization (PSO):** Comportamento de enxame para exploração do espaço de busca.
    3.  ⚡ **Híbrido (GA + Pattern Search):** Exploração global com GA seguida de refinamento local agressivo.
* **Flexível:** Suporta modos de **Maximização** e **Minimização**.
* **Relatórios:** Gera logs em tempo real e um relatório final em Markdown (`relatorio.md`).

## ⚙️ Pré-requisitos

* **Python 3.8** ou superior.
* **Bibliotecas:** O código utiliza apenas bibliotecas padrão do Python (`argparse`, `subprocess`, `random`, `math`, etc.), portanto, **não é necessário instalar dependências via pip**.
* **O Executável Alvo:** Você precisa ter o arquivo `.exe` (ou binário Linux) que deseja otimizar.

## 🔌 Protocolo de Comunicação (Como preparar seu Executável)

Para que este otimizador funcione com o seu programa, seu executável (`simulado.exe`, por exemplo) deve obedecer ao seguinte contrato:

1.  **Entrada:** Deve aceitar **10 argumentos inteiros** via linha de comando.
    ```bash
    ./seu_programa.exe 10 500 30 999 50 60 70 80 90 100
    ```

2.  **Saída:** Deve imprimir **apenas o valor do resultado (score)** na primeira linha da saída padrão (`stdout`).
    ```text
    98.55
    ```
    *(Qualquer outra saída após a primeira linha será ignorada, mas erros de execução farão a avaliação ser descartada).*

## 🚀 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Uso Básico (Maximização):**
    Este comando rodará a otimização por 1 hora (3600 segundos), buscando o maior valor possível.
    ```bash
    python auto_tuning_aprimorado.py --mode max --time 3600 --exe simulado.exe
    ```

3.  **Uso para Minimização:**
    Caso seu objetivo seja diminuir um valor (ex: tempo de execução, erro, custo):
    ```bash
    python auto_tuning_aprimorado.py --mode min --time 1800 --exe ./bin/meu_solver.exe
    ```

### Argumentos Disponíveis

| Argumento | Padrão | Descrição |
| :--- | :--- | :--- |
| `--mode` | `max` | Define o objetivo: `max` (maior valor) ou `min` (menor valor). |
| `--time` | `3600` | Tempo total de execução (em segundos). Esse tempo é dividido igualmente entre as 3 estratégias. |
| `--exe` | `simulado.exe` | Caminho relativo ou absoluto para o executável que será testado. |
| `--seed` | `42` | Semente para geração de números aleatórios (garante reprodutibilidade). |

## 🧪 Testando sem um executável real

Se você quiser testar o script mas ainda não tem o executável pronto, pode criar um script Python simples (`simulado.py`) para agir como o executável.

1.  Crie o arquivo `simulado.py`:
    ```python
    # Exemplo de simulado.py
    import sys
   
    # Pega os 10 argumentos passados pelo otimizador
    args = [int(x) for x in sys.argv[1:]]
   
    # Função objetivo fictícia (ex: soma de todos os parâmetros)
    result = sum(args) 
   
    # Imprime o resultado
    print(result)
    ```

2.  Para rodar o otimizador usando esse script Python como "executável":
    * **Windows:** Crie um `.bat` ou compile com pyinstaller.
    * **Linux/Mac:** Adicione `#!/usr/bin/env python3` no topo e dê permissão `chmod +x`.
    * **Truque rápido:** Você pode alterar a chamada no código principal `cmd = [self.executable_path]...` para `cmd = ["python", self.executable_path]...` se quiser testar scripts `.py` diretamente.

## 📊 Estrutura dos Resultados

Ao final da execução, o script gera:

1.  **Console:** Logs de "NOVO MELHOR" sempre que uma solução superior é encontrada.
2.  **`relatorio.md`:** Um arquivo contendo:
    * Comparativo de performance entre Genético, PSO e Híbrido.
    * A melhor combinação de parâmetros encontrada.
    * O melhor valor objetivo atingido.

## 🛠 Customização

Para alterar a quantidade de parâmetros ou os limites (atualmente 1 a 1000), edite a constante `BOUNDS` no início do arquivo `auto_tuning_aprimorado.py`:

```python
# Exemplo: Alterar para 5 parâmetros entre 1 e 100
BOUNDS = [(1, 100)] * 5
