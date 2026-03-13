导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

model=joblib.load('xgboost.pkl')
test_dataset=pd.read_csv('test_dataset.csv')
feature_names=[
"bnp_total", "hospital_day", "sbp_baseline", "adl_total", "sbp_admit", "age", "pre_apt", "mono_total", "anc_total", "agitation", "post_gastric_tube", "crp_total"

]



# ====================== 2. Streamlit页面配置 ======================
st.set_page_config(page_title="急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器", layout="wide")
st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取急性缺血性脑卒中血管内治疗术后症状性出血转化风险评估结果")

# ====================== 3. 特征输入组件（按编码规则设计） ======================
# 1. bnp_total（数值型：输入具体小时数，自动编码为0=正常/1=异常）
bnp_total_num = st.number_input(
    "基线BNP",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)

# 2. hospital_day（数值型：输入具体小时数，自动编码为0=正常/1=异常）
hospital_day_num = st.number_input(
    "住院天数",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)

# 3. sbp_baseline（数值型：输入具体小时数，自动编码为0=正常/1=异常）
sbp_baseline_num = st.number_input(
    "基线收缩压",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)

# 4. adl_total（数值型：输入具体小时数，自动编码为0=正常/1=异常）
adl_total_num = st.number_input(
    "基线自理能力评分",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)

# 5. sbp_admit（数值型：输入具体小时数，自动编码为0=正常/1=异常）
sbp_admit_num = st.number_input(
    "入院收缩压",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)

# 6. age（数值型：输入具体小时数，自动编码为0=正常/1=异常）
age_num = st.number_input(
    "年龄",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)

# 7. pre_apt（0：否，1：是）
pre_apt = st.selectbox(
    "术前是否使用抗凝抗板药物",
    options=[0, 1],
    format_func=lambda x: "是" if x == 0 else "否")

# 8. mono_total（数值型：输入具体小时数，自动编码为0=正常/1=异常）
mono_total_num = st.number_input(
    "基线单核细胞计数",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)

# 9. anc_total（数值型：输入具体小时数，自动编码为0=正常/1=异常）
anc_total_num = st.number_input(
    "基线中性粒细胞计数",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)


# 10. agitation（多分类：0=无/1=轻度躁动/2=中度躁动/3=重度躁动）
agitation = st.selectbox(
    "术后躁动情况？",
    options=[0, 1, 2, 3],  # 多分类数字编码
    # 自定义显示文本（编码→中文含义）
    format_func=lambda x: {
        0: "无",
        1: "轻度躁动",
        2: "中度躁动",
        3: "重度躁动"
    }[x]
)

# 11. post_gastric_tube（0：否，1：是）
post_gastric_tube = st.selectbox(
    "术后是否留置胃管",
    options=[0, 1],
    format_func=lambda x: "是" if x == 0 else "否"）

# 12. crp_total（数值型：输入具体小时数，自动编码为0=正常/1=异常）
crp_total_num = st.number_input(
    "基线超敏C反应蛋白",
    min_value=0.00,
    max_value=1000000.0,
    value=8.0,
    step=0.5,
    format="%.2f"  # 保留1位小数
)


# ====================== 4. 数据处理与预测 ======================
# 整合用户输入特征
feature_values = [
bnp_total_num,
    hospital_day_num,
    sbp_baseline_num,
    adl_total_num,
    sbp_admit_num,
    age_num,
    pre_apt,               # pre_apt 是 selectbox 的变量名，正确
    mono_total_num,
    anc_total_num,
    agitation,             # 正确
    post_gastric_tube,     # 正确
    crp_total_num
]

# 转换为模型输入格式
if st.button("预测"):
    feature_values = [ ... ]   # 用上面修正后的列表
    features = np.array([feature_values])
    # 然后进行预测
# 预测按钮逻辑
if st.button("预测"):
    # 模型预测
    predicted_class = model.predict(features)[0]  # 0：低风险，1：高风险
    predicted_proba = model.predict_proba(features)[0]  # 概率值
    
    # 显示预测结果（中文适配）
    st.subheader("📊 预测结果")
    risk_label = "高风险" if predicted_class == 1 else "低风险"
    st.write(f"**风险等级：{predicted_class}（{risk_label}）**")
    st.write(f"**风险概率：** 低风险概率 {predicted_proba[0]:.2%} | 高风险概率 {predicted_proba[1]:.2%}")
    
    # 生成个性化建议（中文）
    st.subheader("💡 健康建议")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"模型预测您的症状性出血风险为高风险（概率{probability:.1f}%）。"
            "建议尽快进行全面的评估，重点关注营养摄入、睡眠质量、心理健康等方面，"
            "同时可根据自身情况增加适宜的体育锻炼，改善生活环境。"
        )
        st.write(advice)
    else:
        # 补充低风险的建议（保证逻辑完整）
        advice = (
            f"模型预测您的症状性出血转化风险为低风险（概率{probability:.1f}%）。"
            "建议保持当前的健康生活方式，定期进行常规体检，继续维持良好的饮食、睡眠和运动习惯。"
        )
        st.write(advice)# 导入LIME相关依赖（需提前安装：pip install lime）
import lime
import lime.lime_tabular
import streamlit as st
import streamlit.components.v1 as components

# ====================== 6. LIME解释（适配业务特征） ======================
st.subheader("🔍 LIME特征贡献解释")

# 初始化LIME表格解释器
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=test_dataset.values,  # 测试集特征数据（用于拟合解释器）
    feature_names=feature_names,  # 特征名称列表（需与输入特征对应）
    class_names=['低症状性出血转化风险', '高症状性出血转化风险'],  # 适配业务类别名称
    mode='classification'  # 分类任务模式
)

# 生成当前输入样本的LIME解释
lime_exp = lime_explainer.explain_instance(
    data_row=features.flatten(),  # 展平输入特征数组（适配LIME输入格式）
    predict_fn=model.predict_proba,  # 模型的概率预测函数
    num_features=12  # 显示贡献度前10的重要特征
)

# 以HTML格式展示LIME解释结果（支持表格和可视化）
lime_html = lime_exp.as_html(show_table=True)
components.html(lime_html, height=600, scrolling=True)