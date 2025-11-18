import subprocess
import time
import random
import numpy as np
from scipy.optimize import minimize
from pyswarm import pso
import json
import os 
import sys 

# --- Configurações Fixas do Modelo ---
EXE_PATH = "modelo.exe" 
TIME_LIMIT = 3600 # 1 hora em segundos
LOG_FILE = "resultados_autotuning.json"
CATEGORICAL_P1 = ['baixo', 'medio', 'alto'] 

# Limites e Bounds
BOUNDS_P_INT = (1, 100)
BOUNDS = [(0, 2)] + [(BOUNDS_P_INT[0], BOUNDS_P_INT[1])] * 9

# Variáveis globais (serão definidas pelo menu)
CONFIG_MODE = 'min'  
SELECTED_STRATEGY = '' 
start_time = time.time()
best_overall_result = float('inf') 
best_overall_params = None

# --- Mapeamento das Estratégias ---
STRATEGIES_MAP = {
    '1': ('SIMPLEX (Nelder-Mead)', 'simplex'),
    '2': ('Pattern Search (Powell)', 'patternsearch'),
    '3': ('Combinada (PSO + SIMPLEX)', 'pso_simplex'),
}

# -------------------------------------------------------------------
# FUNÇÕES DE SUPORTE (objective_wrapper, encode_params, evaluate_exe, 
# check_time_and_log, initialize_log)
# MANTÊM-SE AS MESMAS
# -------------------------------------------------------------------

def objective_wrapper(result):
    """Retorna o valor a ser minimizado, dependendo do CONFIG_MODE."""
    if CONFIG_MODE == 'min':
        return result
    else: 
        return -result

def encode_params(x_vector):
    """Decodifica o vetor de otimização contínuo em parâmetros do modelo."""
    p1_index = int(round(x_vector[0]))
    p1 = CATEGORICAL_P1[min(p1_index, len(CATEGORICAL_P1) - 1)]
    p_ints = [max(BOUNDS_P_INT[0], min(BOUNDS_P_INT[1], int(round(val)))) 
              for val in x_vector[1:]]
    params = [p1] + p_ints
    return [str(p) for p in params]

def evaluate_exe(x_vector):
    """Função objetivo que executa o programa .exe e retorna o resultado."""
    params_str = encode_params(x_vector)
    try:
        command = [EXE_PATH] + params_str
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30 
        )
        model_output = float(result.stdout.strip())
        return objective_wrapper(model_output)
    except subprocess.CalledProcessError as e:
        # print(f"Erro na execução do EXE: {e.stderr.strip()}")
        return 1e9 if CONFIG_MODE == 'min' else -1e9
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        # print(f"Erro de processamento/timeout: {e}") 
        return 1e9 if CONFIG_MODE == 'min' else -1e9

def check_time_and_log(current_result, current_x, strategy_name):
    """Verifica se o tempo limite foi atingido e atualiza o melhor resultado global."""
    global start_time, best_overall_result, best_overall_params, CONFIG_MODE
    
    elapsed_time = time.time() - start_time
    original_result = current_result if CONFIG_MODE == 'min' else -current_result
    
    is_better = False
    if CONFIG_MODE == 'min' and original_result < best_overall_result:
        is_better = True
    elif CONFIG_MODE == 'max' and original_result > best_overall_result:
        is_better = True

    if is_better or np.isinf(best_overall_result) or best_overall_params is None: 
        best_overall_result = original_result
        best_overall_params = encode_params(current_x)
        
        log_entry = {
            "time_s": round(elapsed_time, 2),
            "strategy": strategy_name,
            "mode": CONFIG_MODE,
            "result": best_overall_result,
            "parameters": best_overall_params
        }
        
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(log_entry) + ',\n')
            print(f"[{strategy_name}] NOVO MELHOR: {CONFIG_MODE}={best_overall_result:.4f} em {elapsed_time:.2f}s")
        except Exception as e:
            print(f"Erro ao escrever no log: {e}")

    if elapsed_time > TIME_LIMIT:
        print(f"\n[FIM] Tempo limite de {TIME_LIMIT}s atingido. Encerrando otimização.")
        raise Exception("Time Limit Reached")

    def tracked_objective(x):
        try:
            f_x = evaluate_exe(x)
            check_time_and_log(f_x, x, strategy_name)
            return f_x
        except Exception as e:
            if str(e) == "Time Limit Reached":
                raise StopIteration
            raise e 

    return tracked_objective

def initialize_log():
    """Inicializa o log com informações de cabeçalho."""
    global CONFIG_MODE, EXE_PATH, TIME_LIMIT
    try:
        with open(LOG_FILE, 'w') as f:
            f.write(f'[\n{{"info": "Autotuning log para {EXE_PATH}", "mode": "{CONFIG_MODE}", "limit_s": {TIME_LIMIT}}},\n')
    except Exception as e:
        print(f"Erro ao inicializar o log: {e}")

# -------------------------------------------------------------------
# ESTRATÉGIAS DE OTIMIZAÇÃO (SEM ALTERAÇÕES)
# -------------------------------------------------------------------

def strategy_simplex():
    print("--- 1. Estratégia: SIMPLEX (Nelder-Mead) ---")
    x0 = np.array([1] + [50] * 9)
    try:
        tracked_obj = check_time_and_log(best_overall_result, x0, "SIMPLEX") 
        minimize(tracked_obj, x0, method='Nelder-Mead', options={'maxiter': 1000, 'fatol': 1e-4})
    except StopIteration:
        print("SIMPLEX interrompido pelo limite de tempo.")
        
def strategy_pattern_search():
    print("--- 2. Estratégia: Pattern Search (Powell) ---")
    x0 = np.array([random.uniform(b[0], b[1]) for b in BOUNDS])
    try:
        tracked_obj = check_time_and_log(best_overall_result, x0, "PatternSearch")
        minimize(tracked_obj, x0, method='Powell', bounds=BOUNDS, options={'maxiter': 500})
    except StopIteration:
        print("Pattern Search interrompido pelo limite de tempo.")

def strategy_pso_simplex():
    print("--- 3. Estratégia Combinada: PSO Global + SIMPLEX Local ---")
    lb = np.array([b[0] for b in BOUNDS])
    ub = np.array([b[1] for b in BOUNDS])
    
    try:
        # Etapa Global: PSO
        tracked_obj = check_time_and_log(best_overall_result, lb, "PSO_Global")
        xopt, fopt = pso(tracked_obj, lb, ub, swarmsize=10, maxiter=100, minfunc=1e-4, debug=False)
        print(f"PSO Global finalizado. Melhor ponto encontrado: {xopt}")
        
        # Etapa Local: SIMPLEX
        print("--- 3. Etapa Local: SIMPLEX Refinamento ---")
        x0_local = xopt
        tracked_obj_local = check_time_and_log(fopt, x0_local, "SIMPLEX_Refinement") 
        minimize(tracked_obj_local, x0_local, method='Nelder-Mead', options={'maxiter': 200, 'fatol': 1e-5})

    except StopIteration:
        print("Estratégia Combinada interrompida pelo limite de tempo.")

# -------------------------------------------------------------------
# FUNÇÃO DE INTERFACE DE MENU (ATUALIZADA)
# -------------------------------------------------------------------

def display_menu():
    """Exibe o menu de seleção e captura as escolhas do usuário."""
    global CONFIG_MODE, SELECTED_STRATEGY, best_overall_result

    # 1. Limpa a tela
    if os.name == 'nt': 
        os.system('cls')
    else: 
        os.system('clear')

    print("=========================================")
    print("    ⚙️ AUTOTUNING DE MODELO MATEMÁTICO    ")
    print("=========================================")
    print(f"Modelo: {EXE_PATH} | Limite de Tempo: {TIME_LIMIT/3600:.0f} hora(s)")
    print("-" * 39)

    # 2. Seleção do Modo (MIN ou MAX) com números
    print("\n## 🎯 1. ESCOLHA DO OBJETIVO")
    print("1. MINIMIZAR o resultado (buscar o valor mais baixo)")
    print("2. MAXIMIZAR o resultado (buscar o valor mais alto)")
    
    while True:
        mode_choice = input("Digite o NÚMERO do objetivo (1 ou 2): ")
        if mode_choice == '1':
            CONFIG_MODE = 'min'
            best_overall_result = float('inf')
            break
        elif mode_choice == '2':
            CONFIG_MODE = 'max'
            best_overall_result = float('-inf')
            break
        else:
            print("Opção inválida. Digite 1 para MINIMIZAR ou 2 para MAXIMIZAR.")
    
    # 3. Seleção da Estratégia
    print("-" * 39)
    print("\n## 📈 2. ESCOLHA DA ESTRATÉGIA DE OTIMIZAÇÃO")
    print("1. SIMPLEX (Nelder-Mead) - Tradicional, Local")
    print("2. Pattern Search (Powell) - Tradicional, Local, sem Gradiente")
    print("3. Combinada (PSO + SIMPLEX) - Global (PSO) + Refinamento Local (SIMPLEX)")
    
    while True:
        strategy_choice = input("Digite o NÚMERO da estratégia (1, 2 ou 3): ")
        if strategy_choice in STRATEGIES_MAP:
            SELECTED_STRATEGY = STRATEGIES_MAP[strategy_choice][1]
            break
        else:
            print("Opção inválida. Escolha 1, 2 ou 3.")

    print(f"\n✅ Configuração Selecionada:")
    print(f"   Objetivo: **{CONFIG_MODE.upper()}**")
    print(f"   Estratégia: **{STRATEGIES_MAP[strategy_choice][0]}**")
    print("-" * 39)
    input("Pressione ENTER para INICIAR o Autotuning...")

def main():
    # 1. Exibe e processa o menu
    display_menu()
    
    # 2. Mapeamento da Estratégia
    strategies_map_func = {
        'simplex': strategy_simplex,
        'patternsearch': strategy_pattern_search,
        'pso_simplex': strategy_pso_simplex,
    }
    target_strategy = strategies_map_func[SELECTED_STRATEGY]
    
    # 3. Inicialização e Execução
    initialize_log()
    print(f"\n🚀 Iniciando Autotuning. Estratégia: {SELECTED_STRATEGY.upper()}. Modo: {CONFIG_MODE.upper()}.")
    print("--- Lembrete: Se houver [WinError 2], o 'modelo.exe' não foi encontrado na pasta atual. ---")
    
    try:
        global start_time
        start_time = time.time() 
        target_strategy()
    except Exception as e:
        if str(e) != "Time Limit Reached":
            print(f"Erro inesperado durante a execução da estratégia: {e}")
    
    # 4. Finalização
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f'{{"Final_Best_Result": {best_overall_result}, "Final_Best_Params": {best_overall_params}}}\n]')
    except Exception as e:
        print(f"Erro ao finalizar o log: {e}")
        
    print(f"\n✅ Otimização Concluída.")
    print(f"Estratégia usada: {SELECTED_STRATEGY.upper()}")
    print(f"Melhor resultado global ({CONFIG_MODE}): {best_overall_result:.4f}")
    print(f"Melhores parâmetros: {best_overall_params}")
    print(f"Detalhes no arquivo: {LOG_FILE}")


if __name__ == "__main__":
    main()

#Auto Tunning

#-2 estratégias tradicionais
#-1 combinada com pelo menos duas estratégias
#-configurador (max ou min)
#-README com instruções
#-Apresentação/Relatório

#Estratégias
#-Pattern Searsh
#-Particle Swarm
#-Algoritmo  genético
#-SIMPLEX