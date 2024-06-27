import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from datetime import datetime

st.title("哈啰自动化A/B test Demo")

st.header("实验设计")
ownership = st.text_input("Owner")
unit_of_diversion = st.selectbox("分流单元", ["User", "Session", "Region"])
key_metrics = st.text_area("核心指标")

data_loaded = False

if st.button("数据导入"):
    st.write("数据导入功能在demo中不可用")
    data_loaded = False

if st.button("生成demo模拟数据"):
    def simulate_data(size=1000):
        control = np.random.normal(loc=50, scale=5, size=size)
        treatment = np.random.normal(loc=55, scale=5, size=size)
        return control, treatment

    control, treatment = simulate_data()
    data_loaded = True

# Automated timeline data
start_date = datetime(2024, 4, 10)
end_date = datetime(2024, 6, 27)
phases = [
    (datetime(2024, 4, 10), datetime(2024, 4, 24), "Phase 1"),
    (datetime(2024, 4, 24), datetime(2024, 6, 10), "Phase 2"),
    (datetime(2024, 6, 10), datetime(2024, 6, 27), "Phase 3")
]

if data_loaded:
    # Statistical Test
    t_stat, p_value = ttest_ind(control, treatment)

    # Experiment Outcomes
    st.header("实验结论")
    st.write(f"本实验的p值为： {p_value:.4f}")
    if p_value < 0.05:
        st.write("\n\n由于p值小于显著性水平0.05，我们拒绝了原假设。这表明控制组和实验组之间存在显著的统计学差异。")
        st.write("\n根据这些结果，建议考虑实施在实验组中测试的更改，因为它们显示出了显著的影响。")
    else:
        st.write("\n\n由于p值大于显著性水平0.05，我们未能拒绝原假设。这表明控制组和处理组之间没有显著的统计学差异。")
        st.write("\n根据这些结果，不建议实施在实验组中测试的更改，因为它们未显示出显著的影响。")

    # 实验信息
    st.header("实验信息")
    st.write(f"Timeline: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    st.write(f"Mean (Control): {np.mean(control)}")
    st.write(f"Mean (Treatment): {np.mean(treatment)}")
    st.write(f"T-statistic: {t_stat}, P-value: {p_value}")

    # 画图
    fig, ax = plt.subplots()
    ax.hist(control, alpha=0.5, label='Control')
    ax.hist(treatment, alpha=0.5, label='Treatment')
    ax.legend(loc='upper right')
    st.pyplot(fig)

 
    # timeline
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot([start_date, end_date], [0, 0], color='blue', linewidth=2)
    for phase_start, phase_end, phase_name in phases:
        ax.plot([phase_start, phase_end], [0, 0], linewidth=6, label=phase_name)
        ax.text((phase_start + (phase_end - phase_start) / 2), 0.1, phase_name, rotation=45, verticalalignment='bottom')

    ax.set_yticks([])
    ax.set_xticks([start_date] + [phase_end for _, phase_end, _ in phases])
    ax.set_xticklabels([start_date.strftime('%Y-%m-%d')] + [phase_end.strftime('%Y-%m-%d') for _, phase_end, _ in phases])
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(-1, 1)
    st.pyplot(fig)


    # 生成报告
    report = f"""
    ### Experiment Report

    **Ownership:** {ownership}
    **Unit of Diversion:** {unit_of_diversion}
    **Key Metrics:** {key_metrics}

    **Timeline:** {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}

    **Results:**
    - Mean (Control): {np.mean(control)}
    - Mean (Treatment): {np.mean(treatment)}
    - T-statistic: {t_stat}
    - P-value: {p_value}

    **Conclusion:**
    The treatment group showed a statistically significant improvement over the control group with a p-value of {p_value}.
    """

    if p_value < 0.05:
        report += "\n\n由于 p 值小于显著性水平 0.05，我们拒绝了原假设。这表明控制组和处理组之间存在显著的统计学差异。"
        report += "\n根据这些结果，建议考虑实施在处理组中测试的更改，因为它们显示出了显著的影响。"
    else:
        report += "\n\n由于 p 值大于显著性水平 0.05，我们未能拒绝原假设。这表明控制组和处理组之间没有显著的统计学差异。"
        report += "\n根据这些结果，不建议实施在处理组中测试的更改，因为它们未显示出显著的影响。"
    st.download_button("Download Report", data=report, file_name='experiment_report.txt')

else:
    st.write("请载入数据/生成demo模拟数据")