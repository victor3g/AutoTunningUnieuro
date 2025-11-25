# auto_tuning.py
import argparse
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

# ----------------------------
# Configurações gerais
# ----------------------------

BOUNDS = [(1, 100)] * 5  # 5 parâmetros inteiros
DEFAULT_TOTAL_SECONDS = 3600  # 1 hora padrão
EXECUTABLE_PATH = "simulado.exe"  # altere se necessário
SEED = 42  # reprodutibilidade

random.seed(SEED)


# ----------------------------
# Utilidades
# ----------------------------

def clamp_int(x: int, low: int, high: int) -> int:
    return max(low, min(high, x))


def clamp_vec_int(vec: List[int], bounds: List[Tuple[int, int]]) -> List[int]:
    return [clamp_int(int(round(v)), low, high) for v, (low, high) in zip(vec, bounds)]


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ----------------------------
# Avaliação do executável
# ----------------------------

class ExecutableEvaluator:
    """
    Avalia o executável `simulado.exe` com 5 inteiros como parâmetros.
    Espera que o executável imprima uma única linha com o valor objetivo (float).
    Exemplo de chamada: simulado.exe 10 20 30 40 50
    """

    def __init__(self, executable_path: str, timeout_sec: int = 30):
        self.executable_path = executable_path
        self.timeout_sec = timeout_sec

    def evaluate(self, params: List[int]) -> Optional[float]:
        cmd = [self.executable_path] + [str(p) for p in params]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )
            if result.returncode != 0:
                # Se o executável retorna erro, descartamos esta avaliação
                return None
            stdout = result.stdout.strip()
            # Tenta converter a primeira linha como float
            line = stdout.splitlines()[0] if stdout else ""
            value = float(line)
            return value
        except Exception:
            return None


# ----------------------------
# Modo de otimização (max/min)
# ----------------------------

class ObjectiveMode:
    def __init__(self, mode: str):
        if mode not in ("max", "min"):
            raise ValueError("mode deve ser 'max' ou 'min'")
        self.mode = mode

    def better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def worst_value(self) -> float:
        return -math.inf if self.mode == "max" else math.inf

    def sign(self) -> float:
        return 1.0 if self.mode == "max" else -1.0


# ----------------------------
# Base de estratégia
# ----------------------------

@dataclass
class Result:
    name: str
    best_params: List[int]
    best_value: Optional[float]
    evaluations: int
    start_time: str
    end_time: str
    notes: str


class Strategy:
    def __init__(self, name: str, evaluator: ExecutableEvaluator, bounds: List[Tuple[int, int]], mode: ObjectiveMode):
        self.name = name
        self.evaluator = evaluator
        self.bounds = bounds
        self.mode = mode

    def run(self, seconds_budget: int) -> Result:
        raise NotImplementedError


# ----------------------------
# Estratégia: Algoritmo Genético (GA)
# ----------------------------

class GeneticAlgorithm(Strategy):
    def __init__(
        self,
        evaluator: ExecutableEvaluator,
        bounds: List[Tuple[int, int]],
        mode: ObjectiveMode,
        pop_size: int = 40,
        tournament_k: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.15
    ):
        super().__init__("Algoritmo Genético", evaluator, bounds, mode)
        self.pop_size = pop_size
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _random_individual(self) -> List[int]:
        return [random.randint(low, high) for (low, high) in self.bounds]

    def _evaluate_population(self, pop: List[List[int]]) -> List[Tuple[List[int], Optional[float]]]:
        return [(ind, self.evaluator.evaluate(ind)) for ind in pop]

    def _tournament_select(self, evaluated_pop: List[Tuple[List[int], Optional[float]]]) -> List[int]:
        candidates = random.sample(evaluated_pop, self.tournament_k)
        candidates = [c for c in candidates if c[1] is not None]
        if not candidates:
            # se todos None, escolhe aleatório bruto
            return random.choice(evaluated_pop)[0]
        best = candidates[0]
        for cand in candidates[1:]:
            if self.mode.better(cand[1], best[1]):
                best = cand
        return best[0]

    def _crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.crossover_rate:
            return p1[:], p2[:]
        point = random.randint(1, len(p1) - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    def _mutate(self, ind: List[int]) -> List[int]:
        for i, (low, high) in enumerate(self.bounds):
            if random.random() < self.mutation_rate:
                step = random.randint(-5, 5)
                ind[i] = clamp_int(ind[i] + step, low, high)
        return ind

    def run(self, seconds_budget: int) -> Result:
        start = time.time()
        end_time = start + seconds_budget

        pop = [self._random_individual() for _ in range(self.pop_size)]
        evaluated = self._evaluate_population(pop)
        eval_count = len(pop)

        # Inicializa melhor
        best_params = pop[0]
        best_value = None
        for ind, val in evaluated:
            if val is None:
                continue
            if best_value is None or self.mode.better(val, best_value):
                best_value = val
                best_params = ind[:]

        # Evolução por tempo
        while time.time() < end_time:
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(evaluated)
                p2 = self._tournament_select(evaluated)
                c1, c2 = self._crossover(p1, p2)
                c1 = clamp_vec_int(self._mutate(c1), self.bounds)
                c2 = clamp_vec_int(self._mutate(c2), self.bounds)
                new_pop.extend([c1, c2])
            new_pop = new_pop[:self.pop_size]
            evaluated = self._evaluate_population(new_pop)
            eval_count += len(new_pop)
            for ind, val in evaluated:
                if val is None:
                    continue
                if best_value is None or self.mode.better(val, best_value):
                    best_value = val
                    best_params = ind[:]

        return Result(
            name=self.name,
            best_params=best_params,
            best_value=best_value,
            evaluations=eval_count,
            start_time=now_str(),
            end_time=now_str(),
            notes=f"População {self.pop_size}, crossover {self.crossover_rate}, mutação {self.mutation_rate}"
        )


# ----------------------------
# Estratégia: Enxame de Partículas (PSO)
# ----------------------------

class ParticleSwarm(Strategy):
    def __init__(
        self,
        evaluator: ExecutableEvaluator,
        bounds: List[Tuple[int, int]],
        mode: ObjectiveMode,
        swarm_size: int = 40,
        w: float = 0.7,
        c1: float = 1.6,
        c2: float = 1.6
    ):
        super().__init__("Particle Swarm", evaluator, bounds, mode)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def run(self, seconds_budget: int) -> Result:
        start = time.time()
        end_time = start + seconds_budget

        dim = len(self.bounds)
        # Inicialização
        positions = [[random.randint(low, high) for (low, high) in self.bounds] for _ in range(self.swarm_size)]
        velocities = [[random.uniform(-10, 10) for _ in range(dim)] for _ in range(self.swarm_size)]
        pbest_pos = [p[:] for p in positions]
        pbest_val: List[Optional[float]] = [None] * self.swarm_size

        eval_count = 0
        gbest_pos = positions[0][:]
        gbest_val: Optional[float] = None

        # Avaliar inicial
        for i, pos in enumerate(positions):
            val = self.evaluator.evaluate(pos)
            eval_count += 1
            pbest_val[i] = val
            if val is not None and (gbest_val is None or self.mode.better(val, gbest_val)):
                gbest_val = val
                gbest_pos = pos[:]

        # Iterações por tempo
        while time.time() < end_time:
            for i in range(self.swarm_size):
                r1 = random.random()
                r2 = random.random()
                for d in range(dim):
                    velocities[i][d] = (
                        self.w * velocities[i][d]
                        + self.c1 * r1 * (pbest_pos[i][d] - positions[i][d])
                        + self.c2 * r2 * (gbest_pos[d] - positions[i][d])
                    )
                    positions[i][d] = positions[i][d] + velocities[i][d]
                positions[i] = clamp_vec_int(positions[i], self.bounds)
                # Avaliar
                val = self.evaluator.evaluate(positions[i])
                eval_count += 1
                # Atualizar pbest
                if val is not None:
                    if pbest_val[i] is None or self.mode.better(val, pbest_val[i]):
                        pbest_val[i] = val
                        pbest_pos[i] = positions[i][:]
                    # Atualizar gbest
                    if gbest_val is None or self.mode.better(val, gbest_val):
                        gbest_val = val
                        gbest_pos = positions[i][:]

        return Result(
            name=self.name,
            best_params=gbest_pos,
            best_value=gbest_val,
            evaluations=eval_count,
            start_time=now_str(),
            end_time=now_str(),
            notes=f"Swarm {self.swarm_size}, w={self.w}, c1={self.c1}, c2={self.c2}"
        )


# ----------------------------
# Estratégia: Pattern Search (busca por padrões)
# ----------------------------

class PatternSearch(Strategy):
    def __init__(
        self,
        evaluator: ExecutableEvaluator,
        bounds: List[Tuple[int, int]],
        mode: ObjectiveMode,
        init_step: int = 10,
        min_step: int = 1
    ):
        super().__init__("Pattern Search", evaluator, bounds, mode)
        self.init_step = init_step
        self.min_step = min_step

    def run_local(self, start_params: List[int], seconds_budget: int) -> Tuple[List[int], Optional[float], int]:
        start = time.time()
        end_time = start + seconds_budget
        current = start_params[:]
        current_val = self.evaluator.evaluate(current)
        eval_count = 1
        step = self.init_step

        dim = len(self.bounds)

        while time.time() < end_time and step >= self.min_step:
            improved = False
            # Vizinhança de padrões: +/- step em cada dimensão
            neighbors = []
            for d in range(dim):
                for delta in (-step, step):
                    cand = current[:]
                    cand[d] = clamp_int(cand[d] + delta, self.bounds[d][0], self.bounds[d][1])
                    neighbors.append(cand)
            # Avalia vizinhos
            for cand in neighbors:
                val = self.evaluator.evaluate(cand)
                eval_count += 1
                if val is None:
                    continue
                if current_val is None or self.mode.better(val, current_val):
                    current = cand[:]
                    current_val = val
                    improved = True
            # Ajusta passo
            if improved:
                # mantém passo
                pass
            else:
                step = max(self.min_step, step // 2)

        return current, current_val, eval_count

    def run(self, seconds_budget: int) -> Result:
        # Inicialização aleatória seguida de busca local
        start_point = [random.randint(low, high) for (low, high) in self.bounds]
        best_params, best_value, eval_count = self.run_local(start_point, seconds_budget)
        return Result(
            name=self.name,
            best_params=best_params,
            best_value=best_value,
            evaluations=eval_count,
            start_time=now_str(),
            end_time=now_str(),
            notes=f"Passo inicial {self.init_step}, passo mínimo {self.min_step}"
        )


# ----------------------------
# Estratégia combinada: GA + Pattern Search
# ----------------------------

class HybridGAPlusPattern(Strategy):
    def __init__(
        self,
        evaluator: ExecutableEvaluator,
        bounds: List[Tuple[int, int]],
        mode: ObjectiveMode,
        ga_share: float = 0.7  # 70% do tempo no GA, 30% no refinamento (Pattern Search)
    ):
        super().__init__("Combinada: GA + Pattern Search", evaluator, bounds, mode)
        self.ga_share = ga_share
        self.ga = GeneticAlgorithm(evaluator, bounds, mode)
        self.ps = PatternSearch(evaluator, bounds, mode)

    def run(self, seconds_budget: int) -> Result:
        ga_time = int(seconds_budget * self.ga_share)
        ps_time = max(1, seconds_budget - ga_time)

        ga_res = self.ga.run(ga_time)
        # usa o melhor do GA como ponto inicial para refinamento local
        best_params, best_value, eval_count_ps = self.ps.run_local(ga_res.best_params, ps_time)

        # Escolhe melhor final
        final_params = best_params
        final_value = best_value
        eval_total = ga_res.evaluations + eval_count_ps

        return Result(
            name=self.name,
            best_params=final_params,
            best_value=final_value,
            evaluations=eval_total,
            start_time=now_str(),
            end_time=now_str(),
            notes=f"GA ({ga_time}s) + PS ({ps_time}s). {ga_res.notes}; {self.ps.name}: {self.ps.init_step}->{self.ps.min_step}"
        )


# ----------------------------
# Orquestração e relatório
# ----------------------------

def format_params(params: List[int]) -> str:
    return "(" + ", ".join(str(p) for p in params) + ")"

def generate_report(results: List[Result], mode: ObjectiveMode, total_seconds: int, path: str = "relatorio.md") -> None:
    # Encontra melhor geral
    best_overall = None
    for r in results:
        if r.best_value is None:
            continue
        if best_overall is None or mode.better(r.best_value, best_overall.best_value):
            best_overall = r

    lines = []
    lines.append("# Relatório de apresentação — Auto Tuning\n")
    lines.append(f"**Data/hora:** {now_str()}\n")
    lines.append(f"**Executável:** {EXECUTABLE_PATH}\n")
    lines.append(f"**Parâmetros:** 5 inteiros no intervalo [1, 100]\n")
    lines.append(f"**Modo:** {'Maximização' if mode.mode == 'max' else 'Minimização'}\n")
    lines.append(f"**Tempo total:** {total_seconds} segundos (dividido igualmente entre 3 execuções)\n")
    lines.append("\n---\n")
    lines.append("## Comparação de estratégias\n")
    lines.append("| Estratégia | Melhor parâmetros | Melhor valor | Avaliações | Observações |\n")
    lines.append("|-----------|-------------------|--------------|------------|-------------|\n")
    for r in results:
        val_str = "None" if r.best_value is None else f"{r.best_value:.6f}"
        lines.append(f"| {r.name} | {format_params(r.best_params)} | {val_str} | {r.evaluations} | {r.notes} |\n")

    lines.append("\n---\n")
    lines.append("## Destaques e análise\n")
    if best_overall and best_overall.best_value is not None:
        lines.append(f"- **Melhor estratégia:** {best_overall.name}\n")
        lines.append(f"- **Melhor valor:** {best_overall.best_value:.6f}\n")
        lines.append(f"- **Parâmetros correspondentes:** {format_params(best_overall.best_params)}\n")
    else:
        lines.append("- **Observação:** nenhuma avaliação válida foi retornada pelo executável.\n")

    lines.append("\n---\n")
    lines.append("## Metodologia\n")
    lines.append("- **Algoritmo Genético:** seleção por torneio, crossover de um ponto, mutação discreta com steps aleatórios.\n")
    lines.append("- **Particle Swarm:** atualização de posições com velocidades reais, arredondamento e clamp para inteiros.\n")
    lines.append("- **Pattern Search:** refinamento local por vizinhança com passo adaptativo.\n")
    lines.append("- **Combinada (GA + PS):** exploração global (GA) seguida de refinamento local (PS).\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_readme(path: str = "README.md") -> None:
    lines = []
    lines.append("# Auto Tuning — Instruções\n")
    lines.append("Este projeto realiza busca automática de maximização/minimização de `simulado.exe` com 5 parâmetros inteiros em [1, 100].\n")
    lines.append("## Pré-requisitos\n")
    lines.append("- Python 3.8+\n")
    lines.append("- `simulado.exe` acessível no diretório atual ou informar o caminho via parâmetro `--exe`.\n")
    lines.append("## Como executar\n")
    lines.append("```bash\n")
    lines.append("# Maximização, tempo total 1 hora (3600s), executável padrão ./simulado.exe\n")
    lines.append("python auto_tuning.py --mode max --time 3600\n")
    lines.append("\n# Minimização, tempo total 1800s, especificando caminho do executável\n")
    lines.append("python auto_tuning.py --mode min --time 1800 --exe C:\\\\caminho\\\\simulado.exe\n")
    lines.append("```\n")
    lines.append("## Saídas geradas\n")
    lines.append("- `relatorio.md`: relatório comparando as 3 execuções e destacando a melhor.\n")
    lines.append("- logs no console com progresso resumido.\n")
    lines.append("## Estratégias implementadas\n")
    lines.append("- Algoritmo Genético (tradicional)\n")
    lines.append("- Particle Swarm (tradicional)\n")
    lines.append("- Combinada: GA + Pattern Search (refinamento local)\n")
    lines.append("## Observações\n")
    lines.append("- O executável deve imprimir um único valor objetivo em `stdout` (float) para cada chamada.\n")
    lines.append("- Se o executável usar outro formato de saída, adapte a função `ExecutableEvaluator.evaluate`.\n")
    lines.append("- O orçamento total de tempo é dividido igualmente entre as três execuções.\n")
    lines.append("- Para ambientes com avaliações lentas, considere reduzir tamanhos de população/enxame.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto Tuning para simulado.exe com 5 parâmetros inteiros.")
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"], help="Maximização (max) ou minimização (min).")
    parser.add_argument("--time", type=int, default=DEFAULT_TOTAL_SECONDS, help="Tempo total (segundos) distribuído entre 3 execuções.")
    parser.add_argument("--exe", type=str, default=EXECUTABLE_PATH, help="Caminho para o executável simulado.exe.")
    parser.add_argument("--seed", type=int, default=SEED, help="Semente aleatória.")
    args = parser.parse_args()

    random.seed(args.seed)

    # Verifica executável
    exe_path = args.exe
    if not os.path.isfile(exe_path):
        print(f"[AVISO] Executável não encontrado em: {exe_path}. Ajuste com --exe. Continuando mesmo assim para gerar arquivos.")
    evaluator = ExecutableEvaluator(exe_path)
    mode = ObjectiveMode(args.mode)

    # Divide tempo igualmente: 3 execuções
    per_run = max(3, int(args.time // 3))
    print(f"[INFO] Modo: {args.mode} | Tempo total: {args.time}s | Tempo por execução: {per_run}s")

    # Estratégias
    ga = GeneticAlgorithm(evaluator, BOUNDS, mode)
    pso = ParticleSwarm(evaluator, BOUNDS, mode)
    hybrid = HybridGAPlusPattern(evaluator, BOUNDS, mode)

    results: List[Result] = []

    for strat in [ga, pso, hybrid]:
        print(f"[EXEC] Rodando: {strat.name} por {per_run}s ...")
        res = strat.run(per_run)
        print(f"[RESULT] {strat.name}: params {format_params(res.best_params)} | valor {res.best_value}")
        results.append(res)

    # Relatório
    generate_report(results, mode, args.time)
    generate_readme()

    print("[OK] Relatório gerado em relatorio.md")
    print("[OK] README gerado em README.md")


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
