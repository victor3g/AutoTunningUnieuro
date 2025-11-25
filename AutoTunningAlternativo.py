import subprocess
import random
import time
import itertools
import pandas as pd


EXECUTAVEL = "simulado.exe"
TEMPO_LIMITE = 3600  # 1 hora


# -------------------------------------------------------------
# Função para executar o programa externo e obter o resultado
# -------------------------------------------------------------
def executar_simulacao(parametros):
    """Executa o executável com 5 parâmetros inteiros e retorna o valor obtido."""
    cmd = [EXECUTAVEL] + list(map(str, parametros))

    try:
        retorno = subprocess.check_output(cmd, text=True).strip()
        return float(retorno)  # assume que o executável retorna um número
    except Exception as e:
        print("Erro ao executar o programa:", e)
        return -999999  # valor muito baixo para penalizar falhas


# =============================================================
#                 ESTRATÉGIAS DE OTIMIZAÇÃO
# =============================================================

# -------------------------------------------------------------
# 1) Pattern Search
# -------------------------------------------------------------
def pattern_search():
    inicio = time.time()
    best_params = [random.randint(1, 100) for _ in range(5)]
    best_value = executar_simulacao(best_params)

    print("\n=== Pattern Search | Iniciando ===")

    step = 10

    while time.time() - inicio < TEMPO_LIMITE:

        melhorou = False

        for i in range(5):
            for delta in [-step, step]:
                candidato = best_params.copy()
                candidato[i] = max(1, min(100, candidato[i] + delta))

                valor = executar_simulacao(candidato)

                if valor > best_value:
                    best_value = valor
                    best_params = candidato
                    melhorou = True
                    print(f"[Pattern Search] Novo melhor valor: {valor} com {candidato}")

        if not melhorou:
            step = max(1, step // 2)

        if step == 1 and not melhorou:
            break

    return best_params, best_value


# -------------------------------------------------------------
# 2) Particle Swarm Optimization (PSO)
# -------------------------------------------------------------
def particle_swarm():
    inicio = time.time()

    print("\n=== Particle Swarm | Iniciando ===")

    n_particulas = 10
    particulas = [[random.randint(1, 100) for _ in range(5)] for _ in range(n_particulas)]
    velocidades = [[0]*5 for _ in range(n_particulas)]

    p_best = particulas.copy()
    p_best_value = [executar_simulacao(p) for p in particulas]

    g_best = p_best[p_best_value.index(max(p_best_value))]
    g_best_value = max(p_best_value)

    print(f"[PSO] Começando com gbest = {g_best}, valor = {g_best_value}")

    w, c1, c2 = 0.7, 1.5, 1.5

    while time.time() - inicio < TEMPO_LIMITE:

        for i in range(n_particulas):

            # atualizar velocidades
            for d in range(5):
                r1, r2 = random.random(), random.random()
                velocidades[i][d] = (
                    w * velocidades[i][d]
                    + c1 * r1 * (p_best[i][d] - particulas[i][d])
                    + c2 * r2 * (g_best[d] - particulas[i][d])
                )

            # atualizar posições
            particulas[i] = [
                max(1, min(100, particulas[i][d] + int(velocidades[i][d])))
                for d in range(5)
            ]

            # avaliar
            valor = executar_simulacao(particulas[i])

            # atualizar pbest
            if valor > p_best_value[i]:
                p_best_value[i] = valor
                p_best[i] = particulas[i]

                # atualizar gbest
                if valor > g_best_value:
                    g_best_value = valor
                    g_best = particulas[i]
                    print(f"[PSO] Novo melhor valor: {valor} com {g_best}")

    return g_best, g_best_value


# -------------------------------------------------------------
# 3) Estratégia Combinada (PSO + Algoritmo Genético)
# -------------------------------------------------------------
def combinado_pso_ga():
    inicio = time.time()
    print("\n=== Estratégia Combinada PSO + GA | Iniciando ===")

    # começa com PSO (30 minutos)
    metade_tempo = TEMPO_LIMITE / 2

    print("\n--- Etapa PSO ---")
    melhor_pso, valor_pso = particle_swarm()

    # inicia GA com base no melhor do PSO
    print("\n--- Etapa Algoritmo Genético ---")

    best_params = melhor_pso
    best_value = valor_pso

    populacao = [best_params] + [[random.randint(1, 100) for _ in range(5)] for _ in range(9)]
    inicio_ga = time.time()

    while time.time() - inicio < TEMPO_LIMITE:

        # avaliação
        fitness = [executar_simulacao(ind) for ind in populacao]

        # elitismo
        elite = populacao[fitness.index(max(fitness))]
        elite_value = max(fitness)

        if elite_value > best_value:
            best_value = elite_value
            best_params = elite
            print(f"[Combinado] Novo melhor valor: {best_value} com {best_params}")

        # cruzamento
        nova_pop = [elite]
        while len(nova_pop) < 10:
            p1, p2 = random.sample(populacao, 2)
            corte = random.randint(1, 4)
            filho = p1[:corte] + p2[corte:]
            nova_pop.append(filho)

        # mutação
        for ind in nova_pop:
            if random.random() < 0.3:
                i = random.randint(0, 4)
                ind[i] = random.randint(1, 100)

        populacao = nova_pop

    return best_params, best_value


# =============================================================
#             EXECUÇÃO DAS 3 REGRAS + TABELA FINAL
# =============================================================
if __name__ == "__main__":
    resultados = {}

    params_ps, val_ps = pattern_search()
    resultados["Pattern Search"] = (params_ps, val_ps)

    params_pso, val_pso = particle_swarm()
    resultados["Particle Swarm"] = (params_pso, val_pso)

    params_comb, val_comb = combinado_pso_ga()
    resultados["Combinado PSO+GA"] = (params_comb, val_comb)

    print("\n\n================ TABELA FINAL ==================\n")
    tabela = pd.DataFrame({
        "Estratégia": resultados.keys(),
        "Melhor Resultado": [v[1] for v in resultados.values()],
        "Parâmetros": [v[0] for v in resultados.values()]
    })

    print(tabela.to_string(index=False))
