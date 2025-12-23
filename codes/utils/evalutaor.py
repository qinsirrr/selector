import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import json

# 配置中文显示
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


def determine_threshold(val_rank_path, val_confidence_path,target_risks):
    """
    根据验证集确定阈值gamma
    输入: val_data_path - 验证集数据路径（包含按batch存储的softmax和rank）
          target_risks - 目标风险列表（1-H@1）
    输出: 最优阈值及对应的指标
    """
    # 加载验证集数据（假设每个batch包含'softmax'和'rank'键）
    with open(val_rank_path, 'rb') as f:
        rank_data = pickle.load(f)  # 读取rank排名是个列表
    with open(val_confidence_path,'rb') as f:
        confidence_data = pickle.load(f) #读取模型输出的confidence
    #转为numpy数组
    all_ranks = np.array(rank_data['rank'])
    all_confidence = np.array(confidence_data)
    total_samples = len(all_confidence)
    print(f"验证集总样本数: {total_samples}")

    # 按置信度降序排序（高置信度优先）
    sorted_indices = np.argsort(all_confidence)[::-1]  # 降序索引
    sorted_confidence = all_confidence[sorted_indices]  # 排序后的置信度
    sorted_ranks = all_ranks[sorted_indices]  # 排序后的rank

    cumulative_hits = 0  # 累积正确样本数（rank≤1，即H@1的分子）
    risk_candidates = {risk: [] for risk in target_risks}  # 存储每个目标风险的候选阈值

    # 遍历所有样本，计算累积指标
    for i in range(total_samples):
        # 当前样本是否正确（rank≤1则为H@1命中）
        if sorted_ranks[i] <= 1:
            cumulative_hits += 1

        # 计算当前指标
        included_count = i + 1  # 已纳入的样本数（1-based）
        current_coverage = included_count / total_samples  # 回答率（覆盖率）
        current_hits1 = cumulative_hits / included_count  # 当前H@1
        current_risk = 1 - current_hits1  # 当前风险（1-H@1）

        # 记录满足目标风险的候选阈值
        for risk in target_risks:
            if current_risk <= risk:
                risk_candidates[risk].append({
                    'gamma': sorted_confidence[i],  # 当前样本的置信度作为候选阈值
                    'coverage': current_coverage,
                    'actual_risk': current_risk,
                    'included_count': included_count
                })

        # 进度打印
        if (i + 1) % 1000 == 0 or i == total_samples - 1:
            print(f"已处理{i + 1}/{total_samples}样本，当前风险={current_risk:.4f}，覆盖率={current_coverage:.4f}")

    # 选择每个目标风险的最优阈值（最大覆盖率）
    optimal_thresholds = {}
    for risk in target_risks:
        candidates = risk_candidates[risk]
        if not candidates:
            # 无满足条件的样本，纳入所有样本
            final_hits1 = cumulative_hits / total_samples if total_samples > 0 else 0.0
            optimal_thresholds[risk] = {
                'gamma': sorted_confidence[-1] if total_samples > 0 else 0.0,
                'coverage': 1.0 if total_samples > 0 else 0.0,
                'actual_risk': 1 - final_hits1,
                'status': '无满足条件的样本，纳入全部'
            }
        else:
            # 最后一个候选的覆盖率最大（因为按置信度降序遍历，纳入样本数递增）
            best_candidate = candidates[-1]
            optimal_thresholds[risk] = {
                'gamma': best_candidate['gamma'],
                'coverage': best_candidate['coverage'],
                'actual_risk': best_candidate['actual_risk'],
                'status': '满足风险的最大覆盖率'
            }

    return optimal_thresholds, sorted_confidence, sorted_ranks


def plot_risk_coverage(sorted_confidence, sorted_ranks, target_risks, save_dir):
    """绘制风险-覆盖率曲线，标记最优阈值点"""
    total_samples = len(sorted_confidence)
    cumulative_hits = 0
    coverages = []
    risks = []

    for i in range(total_samples):
        if sorted_ranks[i] <= 1:
            cumulative_hits += 1
        included_count = i + 1
        coverages.append(included_count / total_samples)
        risks.append(1 - (cumulative_hits / included_count))  # 1-H@1

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, risks, 'b-', linewidth=2, label='风险（1-H@1）')
    plt.xlabel('覆盖率（回答率）', fontsize=12)
    plt.ylabel('风险', fontsize=12)
    plt.title('风险与覆盖率关系（按置信度降序纳入样本）', fontsize=14)

    # 标记目标风险线
    for risk in target_risks:
        plt.axhline(y=risk, color='r', linestyle='--', alpha=0.7, label=f'{risk * 100}%风险线')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'risk_coverage_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"风险-覆盖率曲线已保存至: {save_path}")


def evaluate_test(test_rank_path,test_confidence_path,optimal_thresholds):
    """在测试集上评估最优阈值的性能"""
    # 加载测试集数据（同验证集格式）
    with open(test_rank_path, 'rb') as f:
        rank_data = pickle.load(f)  # 读取rank排名是个列表
    with open(test_confidence_path,'rb') as f:
        confidence_data = pickle.load(f) #读取模型输出的confidence
    #转为numpy数组
    all_ranks = np.array(rank_data['rank'])
    all_confidence = np.array(confidence_data)
    total_samples = len(all_confidence)
    print(f"测试集总样本数: {total_samples}")

    test_results = {}
    for risk, info in optimal_thresholds.items():
        gamma = info['gamma']
        # 筛选接受的样本（置信度≥gamma）
        kept_mask = all_confidence >= gamma
        kept_ranks = all_ranks[kept_mask]
        kept_count = np.sum(kept_mask)

        if kept_count == 0:
            test_results[risk] = {
                'gamma': gamma,
                'coverage': 0.0,
                'actual_risk': 1.0,
                'hits1': 0.0
            }
            continue

        # 计算测试集指标
        hits1 = (kept_ranks <= 1).mean()  # H@1
        test_results[risk] = {
            'gamma': gamma,
            'coverage': kept_count / total_samples,
            'actual_risk': 1 - hits1,
            'hits1': hits1
        }

    return test_results


def main(val_rank_path, test_rank_path,val_confidence_path,test_confidence_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 目标风险（1-H@1），可根据需求调整
    target_risks = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%

    # 步骤1：在验证集上确定最优阈值
    print("=== 开始在验证集上确定最优阈值 ===")
    optimal_thresholds, sorted_confidence, sorted_ranks = determine_threshold(val_rank_path, val_confidence_path,target_risks)

    # 打印验证集结果
    print("\n=== 验证集最优阈值结果 ===")
    for risk in target_risks:
        info = optimal_thresholds[risk]
        print(f"目标风险: {risk * 100}%")
        print(f"阈值gamma: {info['gamma']:.4f}")
        print(f"实际风险: {info['actual_risk'] * 100:.2f}%")
        print(f"覆盖率: {info['coverage'] * 100:.2f}%")
        print(f"状态: {info['status']}\n")

    # 步骤2：绘制风险-覆盖率曲线
    plot_risk_coverage(sorted_confidence, sorted_ranks, target_risks, save_dir)

    # 步骤3：在测试集上评估
    print("=== 开始在测试集上评估 ===")
    test_results = evaluate_test(test_rank_path,test_confidence_path,optimal_thresholds)

    # 打印测试集结果
    print("\n=== 测试集性能结果 ===")
    for risk in target_risks:
        info = test_results[risk]
        print(f"目标风险: {risk * 100}%")
        print(f"使用阈值gamma: {info['gamma']:.4f}")
        print(f"实际风险: {info['actual_risk'] * 100:.2f}%")
        print(f"覆盖率: {info['coverage'] * 100:.2f}%")
        print(f"H@1: {info['hits1'] * 100:.2f}%\n")

    # 保存结果
    results_data = {
        'val_thresholds': optimal_thresholds,
        'test_metrics': test_results
    }
    
    # 保存为pickle文件
    save_path = os.path.join(save_dir, 'evaluation_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"所有结果已保存至: {save_path}")
    
    # 保存为JSON文件
    json_save_path = os.path.join(save_dir, 'evaluation_results.json')
    
    # 转换numpy类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 转换数据并保存
    json_data = convert_numpy_types(results_data)
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"JSON结果已保存至: {json_save_path}")


# 示例运行（请替换为您的实际路径）
if __name__ == "__main__":
    # val_data_path = "../results/WikiDiverse/validation_results.pkl"  # 验证集数据路径（包含softmax和rank）
    # test_data_path = "../results/WikiDiverse/validation_results.pkl"  # 测试集数据路径
    val_rank_path = "/home/jianbo/Code/MIMIC-selector/codes/data/mimic_data/WikiDiverse/validation_data.pkl"
    test_rank_path="/home/jianbo/Code/MIMIC-selector/codes/data/mimic_data/WikiDiverse/test_data.pkl"
    # val_confidence_path = "/home/jianbo/Code/MIMIC-selector/codes/results/model_confidence/max-prob/val_confidence.pkl"
    # test_confidence_path = "/home/jianbo/Code/MIMIC-selector/codes/results/model_confidence/max-prob/test_confidence.pkl"
    val_confidence_path = "../results/model_confidence/selector/WikiDiverse/validation_confidence.pkl"
    test_confidence_path = "../results/model_confidence/selector/WikiDiverse/test_confidence.pkl"
    save_dir = "../results/r@h_results/selector/WikiDiverse/"
    main(val_rank_path, test_rank_path,val_confidence_path,test_confidence_path, save_dir)
    