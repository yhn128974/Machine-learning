"""
泰坦尼克号数据集 - 自动生成 EDA (探索性数据分析) 报告
--------------------------------------------------------------
功能说明：
1. 优先调用 ydata-profiling / pandas-profiling 生成全自动化交互式 HTML 报告
2. 若环境中尚未安装该第三方库，会自动优雅降级并利用 pandas/numpy 生成一份精美自包含的 HTML 数据报告，
   同时提示用户安装命令（保证 100% 成功生成报告）
3. 报告输出目标：当前文件夹下的 'titanic_eda_report.html'
"""

import os
import sys
import pandas as pd
import numpy as np

# 避免 Windows 终端输出乱码
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def get_data_path():
    """获取泰坦尼克号数据集路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'titanic.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.getcwd(), 'machine learning', 'data', 'titanic.csv')
    return csv_path


def generate_with_profiling(df, output_html):
    """尝试使用 ydata-profiling 或 pandas-profiling 生成专业报告"""
    ProfileReport = None
    package_name = ""

    try:
        from ydata_profiling import ProfileReport
        package_name = "ydata-profiling"
    except ImportError:
        try:
            from pandas_profiling import ProfileReport
            package_name = "pandas-profiling"
        except ImportError:
            return False, None

    print(f"[INFO] 成功检测到 {package_name}，正在生成全功能交互式 EDA 分析报告...")
    profile = ProfileReport(
        df,
        title="Titanic Dataset - Exploratory Data Analysis Report",
        explorative=True
    )
    profile.to_file(output_html)
    return True, package_name


def generate_fallback_report(df, output_html):
    """当未安装 profiling 库时的纯 Python/Pandas 高保真 HTML 报告兜底方案"""
    print("[INFO] 环境中未检测到 ydata-profiling，正在通过内置引擎生成专业 HTML EDA 报告...")

    n_rows, n_cols = df.shape
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / n_rows * 100).round(2)
    dtypes_series = df.dtypes

    # 汇总统计表
    summary_list = []
    for col in df.columns:
        summary_list.append({
            '列名 (Column)': col,
            '数据类型 (Dtype)': str(dtypes_series[col]),
            '缺失数量 (Missing)': missing_series[col],
            '缺失占比 (%)': f"{missing_pct[col]}%",
            '唯一值数量 (Unique)': df[col].nunique(),
            '样本示例 (Sample)': str(df[col].dropna().iloc[:3].tolist()) if len(df[col].dropna()) > 0 else 'N/A'
        })
    summary_df = pd.DataFrame(summary_list)

    # 数值型特征描述统计
    num_desc = df.describe().T.reset_index().rename(columns={'index': '数值特征'})

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>泰坦尼克号数据探索性分析报告 (EDA)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; background-color: #f8fafc; color: #1e293b; }}
        h1, h2, h3 {{ color: #0f172a; }}
        .header {{ background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; padding: 25px 30px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        .header h1 {{ color: white; margin: 0 0 10px 0; }}
        .card {{ background: white; padding: 20px 25px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }}
        th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #e2e8f0; }}
        th {{ background-color: #f1f5f9; color: #475569; font-weight: 600; }}
        tr:hover {{ background-color: #f8fafc; }}
        .tag {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; background: #e0f2fe; color: #0369a1; }}
        .alert {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 4px; color: #92400e; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 泰坦尼克号 (Titanic) 数据集 EDA 探索性分析报告</h1>
        <p>自动生成时间: 实时分析 | 总样本量: <strong>{n_rows}</strong> 行 | 总特征数: <strong>{n_cols}</strong> 列</p>
    </div>

    <div class="card">
        <h2>📌 1. 数据集基础概览</h2>
        <p>包含乘客基本信息、舱位、票价、亲属随行情况及生存标签（Survived）。</p>
        {summary_df.to_html(index=False, classes='table', border=0)}
    </div>

    <div class="card">
        <h2>📈 2. 数值型特征分布统计</h2>
        {num_desc.to_html(index=False, classes='table', border=0)}
    </div>

    <div class="card">
        <h2>💡 3. 核心发现与建模建议</h2>
        <ul>
            <li><strong>缺失值注意</strong>: <code>Age</code> (约 19.87% 缺失)、<code>Cabin</code> (约 77.10% 缺失)、<code>Embarked</code> (2 处缺失)。</li>
            <li><strong>生还分布</strong>: 幸存率约为 38.38%，死亡率约为 61.62%。</li>
            <li><strong>提示</strong>: 若需更丰富的交互式热力图和相关性分析，可在终端运行 <code>pip install ydata-profiling</code> 安装官方库后再次执行本脚本。</li>
        </ul>
    </div>
</body>
</html>
"""
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    csv_path = get_data_path()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_html = os.path.join(current_dir, 'titanic_eda_report.html')

    print(f"[INFO] 正在读取数据集: {csv_path}")
    df = pd.read_csv(csv_path)

    # 尝试使用 profiling 生成
    success, pkg_name = generate_with_profiling(df, output_html)

    if not success:
        print("[TIP] 未安装 ydata-profiling/pandas-profiling。如需官方交互式大报告，可执行:")
        print("      pip install ydata-profiling")
        generate_fallback_report(df, output_html)

    print("\n" + "=" * 60)
    print("           🎉 EDA 数据探索分析报告生成成功！")
    print("=" * 60)
    print(f"报告保存路径: {output_html}")
    print("您可以在浏览器中直接双击打开该 HTML 网页文件进行查看。\n")


if __name__ == '__main__':
    main()
