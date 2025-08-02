import math
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

proName = "3M9N-3"
DATA_FILE = Path(__file__).parent / "data2" / f"{proName}.txt"
excel_file_path = f'results/{proName}/pareto_front.xlsx'
solution_to_plot_id = 11

def load_instance(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        content = f.read().replace(";", "")
    data: dict = {}
    exec(content, {}, data)
    return data

data = load_instance(DATA_FILE)

g: int = data["g"]
tag: List[int] = data["ag"]
job_num: int = data["job"]
machine_num: int = data["machine"]
R: List[int] = data["R"]
P: List[List[int]] = data["P"]
E: List[List[int]] = data["E"]
K: int = data["K"]
U: List[int] = data["U"]
W: List[int] = data["W"]

def adjust_start_time(start: float, duration: float, boundaries: List[int]) -> float:
    while True:
        idx = next((i for i in range(len(boundaries) - 1) if boundaries[i] <= start < boundaries[i + 1]), None)
        if idx is None:
            seg_end = math.inf
        else:
            seg_end = boundaries[idx + 1]
        if start + duration <= seg_end:
            return start
        start = seg_end

def schedule_jobs(job_sequence: List[int]):
    S = [[0.0] * job_num for _ in range(machine_num)]
    F = [[0.0] * job_num for _ in range(machine_num)]
    machine_ready = [0.0] * machine_num

    for job in job_sequence:
        for m in range(machine_num):
            prev_machine_finish = 0.0 if m == 0 else F[m - 1][job]
            earliest_start = max(machine_ready[m], prev_machine_finish)
            if m == 0:
                earliest_start = max(earliest_start, R[job])
            
            feasible_start = adjust_start_time(earliest_start, P[m][job], U)
            S[m][job] = feasible_start
            F[m][job] = feasible_start + P[m][job]
            machine_ready[m] = F[m][job]
    return S, F

def get_price_for_time(time: float) -> int:
    if time >= U[-1]:
        return W[-1]
    for i in range(len(W)):
        if U[i] <= time < U[i+1]:
            return W[i]
    return W[-1]

def agent_completion_sum(F_matrix) -> float:
    total = 0.0
    for a in range(g):
        start_idx, end_idx = tag[a], tag[a + 1]
        jobs_of_agent = range(start_idx, end_idx)
        if not list(jobs_of_agent):
            continue
        makespan = max(F_matrix[machine_num - 1][j] for j in jobs_of_agent)
        total += makespan
    return total

try:
    df_completion = pd.read_excel(excel_file_path, sheet_name='completion_times')
    df_start = pd.read_excel(excel_file_path, sheet_name='start_times')
    df_pareto = pd.read_excel(excel_file_path, sheet_name='pareto_front')

    sol_row = df_pareto[df_pareto['solution_id'] == solution_to_plot_id]
    if sol_row.empty:
        raise ValueError(f"在 'pareto_front' sheet中找不到 solution_id 为 {solution_to_plot_id} 的解。")
    
    sequence_str = sol_row['sequence'].iloc[0]
    JOB_SEQUENCE = [int(j.strip()) for j in str(sequence_str).split(',')]

    assert len(JOB_SEQUENCE) == job_num, "从Excel读取的序列长度与工件数量不一致"

    S, F = schedule_jobs(JOB_SEQUENCE)
    TEC = sum(P[m][j] * E[m][j] * get_price_for_time(S[m][j]) for m in range(machine_num) for j in range(job_num))
    AGENT_TIME_SUM = agent_completion_sum(F)
    print(f"基于前沿解{solution_to_plot_id}的初始调度结果")
    print(f"TEC (总能耗): {TEC}")
    print(f"代理完工时间之和: {AGENT_TIME_SUM}")

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.5)

    ax1 = axs[0]
    colors = plt.cm.get_cmap("tab20", job_num)
    y_offsets = [i * 10 for i in range(machine_num)]

    for m in range(machine_num):
        for j in JOB_SEQUENCE:
            start = S[m][j]
            duration = P[m][j]
            if duration > 1e-6:
                ax1.broken_barh([(start, duration)], (y_offsets[m], 8), facecolors=colors(j), edgecolor="black")
                ax1.text(start + duration / 2, y_offsets[m] + 4, str(j), ha="center", va="center", color="white", fontsize=8)

    is_first_boundary_before = True
    for boundary in U:
        if boundary > 0:
            ax1.axvline(x=boundary, color='r', linestyle='--', linewidth=1.2, label='Price Boundary' if is_first_boundary_before else "")
            is_first_boundary_before = False
    
    ax1.set_yticks([y + 4 for y in y_offsets])
    ax1.set_yticklabels([f"Machine {i + 1}" for i in range(machine_num)])
    ax1.set_ylabel("Machine")
    ax1.set_title("Gantt Chart for Job Schedule (Before Right Shift)")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='upper right')

    ax2 = axs[1]
    sol_completion_data = df_completion[df_completion['solution_id'] == solution_to_plot_id]
    sol_start_data = df_start[df_start['solution_id'] == solution_to_plot_id]

    if sol_start_data.empty or sol_completion_data.empty:
        raise ValueError(f"在文件中找不到 solution_id 为 {solution_to_plot_id} 的 'start_times' 或 'completion_times'。")

    c_times = sol_completion_data.drop('solution_id', axis=1).values
    s_times = sol_start_data.drop('solution_id', axis=1).values
    
    num_jobs_after, num_machines_after = s_times.shape

    for job_id in range(num_jobs_after):
        for m_idx in range(num_machines_after):
            start_time = s_times[job_id, m_idx]
            duration = c_times[job_id, m_idx] - start_time
            
            if duration > 1e-6:
                ax2.broken_barh(
                    xranges=[(start_time, duration)],
                    yrange=(y_offsets[m_idx], 8),
                    facecolors=colors(job_id),
                    edgecolor='black'
                )
                ax2.text(
                    x=start_time + duration / 2,
                    y=y_offsets[m_idx] + 4,
                    s=f"{job_id}",
                    ha='center', va='center', color='white', fontsize=8
                )

    is_first_boundary_after = True
    for boundary in U:
        if boundary > 0:
            ax2.axvline(x=boundary, color='red', linestyle='--', linewidth=1.2, label='Price Boundary' if is_first_boundary_after else "")
            is_first_boundary_after = False

    ax2.set_yticks([y + 4 for y in y_offsets])
    ax2.set_yticklabels([f"Machine {i + 1}" for i in range(num_machines_after)])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Machine")
    ax2.set_title(f"Gantt Chart for Job Schedule After Right Shift (Solution ID: {solution_to_plot_id})")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(handles, labels, loc='upper right')

except (FileNotFoundError, ValueError, KeyError) as e:
    print(f"处理或绘图时发生错误: {e}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.text(0.5, 0.5, f"无法生成图表:\n{e}", ha='center', va='center', wrap=True, color='red')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
