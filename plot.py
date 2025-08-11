import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(instance_base_name: str):
    """
    绘制 B&B 算法和进化算法的帕累托前沿对比图。

    Args:
        instance_base_name (str): 实例的基础名称, 例如 '3M7-1'.
                                  代码会根据这个名称自动查找对应的文件。
                                  - B&B: BBresults/{instance_base_name (用J替换-)}.txt
                                  - EA: results/{instance_base_name (用N替换-)}/pareto_front.csv
    """
    print(f"正在为实例 '{instance_base_name}' 生成对比图...")

    bb_instance_name = instance_base_name.replace('-', 'J-')
    ea_instance_name = instance_base_name.replace('-', 'N-')

    bb_file_path = os.path.join('BBresults', f'{bb_instance_name}.txt')
    ea_file_path = os.path.join('results', ea_instance_name, 'pareto_front.xlsx')

    if not os.path.exists(bb_file_path):
        print(f"错误: B&B 结果文件未找到: {bb_file_path}")
        return
    if not os.path.exists(ea_file_path):
        print(f"错误: 进化算法结果文件未找到: {ea_file_path}")
        return

    try:
        tec_vals, tcta_vals = [], []
        with open(bb_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            next(f)  
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tec, tcta = map(int, line.split())
                    tec_vals.append(tec)
                    tcta_vals.append(tcta)
                except ValueError:
                    break
        bb_df = pd.DataFrame({'TEC': tec_vals, 'TCTA': tcta_vals})

        ea_df = pd.read_excel(ea_file_path)

    except Exception as e:
        print(f"读取数据文件时发生错误: {e}")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    bb_sorted = bb_df.sort_values(by='TEC')
    ax.plot(bb_sorted['TEC'], bb_sorted['TCTA'], marker='s', linestyle='-', color='red', label='B&B', markersize=8, markerfacecolor='white', markeredgewidth=1.5)

    ea_sorted = ea_df.sort_values(by='TEC')
    ax.plot(ea_sorted['TEC'], ea_sorted['TCTA'], marker='o', linestyle='--', color='blue', label='MO_BFO', markersize=8, markerfacecolor='white', markeredgewidth=1.5)

    ax.set_title(f'Pareto Front Comparison: {instance_base_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('TEC', fontsize=12)
    ax.set_ylabel('TCTA', fontsize=12)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_dir = 'comparison_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = os.path.join(output_dir, f'comparison_{instance_base_name}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':

    # instances = ['3M7-1', '3M7-2', '3M7-3','3M7-4','3M7-5',
    #              '3M8-1', '3M8-2', '3M8-3','3M8-4','3M8-5',
    #              '3M9-1', '3M9-2', '3M9-3','3M9-4','3M9-5']
    #              '3M10-5', '3M10-11', '3M10-12','3M10-13','3M10-14']
                 
    instances = ['3M10-15']
    # instances = ['3M7-5']

    # instances = ['3M9-1', '3M9-2', '3M9-3', '3M9-4', '3M9-5']
    # instances = ['3M10-1', '3M10-11', '3M10-3', '3M10-4', '3M10-5']
    # instances = ['3M7-3', '3M9-4']
    for instance in instances:
        plot_comparison(instance)
