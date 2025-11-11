import subprocess

# Parâmetros possíveis
levels = ["baixo", "medio", "alto"]  # x1 textual
x2_x3_min, x2_x3_max = 1, 100
x4_x10_min, x4_x10_max = 0, 100

def run_program(params):
    """
    params: [x1_index, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    x1_index: índice em levels
    x2..x3: 1-100
    x4..x10: 0-100
    """
    x1 = levels[params[0]]
    args = [x1] + [str(p) for p in params[1:]]
    
    try:
        result = subprocess.run(['programa.exe'] + args, capture_output=True, text=True)
        output = result.stdout.strip()
        score = float(output)
        return score
    except ValueError:
        print(f"[AVISO] Saída inválida para parâmetros {params}: {output}")
        return float('-inf')
    except Exception as e:
        print(f"[ERRO] Falha ao rodar {params}: {e}")
        return float('-inf')


def pattern_search(x0, step_sizes, max_iter=50):
    x = x0.copy()
    n = len(x)
    
    for it in range(max_iter):
        improved = False
        current_score = run_program(x)
        print(f"Iter {it}, current params: {x}, score: {current_score}")
        
        for i in range(n):
            for delta in [step_sizes[i], -step_sizes[i]]:
                x_new = x.copy()
                x_new[i] += delta
                
                # Respeitar limites
                if i == 0:
                    x_new[i] = max(0, min(x_new[i], len(levels)-1))  # índice de x1
                elif i in [1,2]:
                    x_new[i] = max(x2_x3_min, min(x_new[i], x2_x3_max))
                else:
                    x_new[i] = max(x4_x10_min, min(x_new[i], x4_x10_max))
                
                score_new = run_program(x_new)
                
                if score_new > current_score:
                    x = x_new
                    current_score = score_new
                    improved = True
                    print(f"  → Nova melhora! {x} com score {current_score}")
        
        # Reduz passo se não houver melhoria
        if not improved:
            step_sizes = [max(1, s//2) for s in step_sizes]
            print(f"  → Nenhuma melhora. Reduzindo passos: {step_sizes}")
        
        # Condição de parada
        if all(s == 1 for s in step_sizes):
            print("Passos mínimos atingidos. Parando.")
            break
    
    return x, run_program(x)


# --- Configuração inicial ---
x0 = [1, 50, 50, 50, 50, 50, 50, 50, 50, 50]  # x1_index=medio, inteiros 50
step_sizes = [1, 10, 10, 10, 10, 10, 10, 10, 10, 10]

best_params, best_score = pattern_search(x0, step_sizes)
print("\n=== RESULTADO FINAL ===")
print("Melhores parâmetros:", [levels[best_params[0]]] + best_params[1:])
print("Melhor score:", best_score)
