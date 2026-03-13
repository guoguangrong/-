# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# 加载模型和数据
model = joblib.load('xgboost_12.pkl')          # 12特征的XGBoost模型
test_dataset = pd.read_csv('test_dataset.csv') # 用于LIME解释的数据集（应包含相同的12个特征列）

# 模型使用的12个特征（必须与训练时的顺序一致）
feature_names = [
    "bnp_total", "hospital_day", "sbp_baseline", "adl_total", "sbp_admit",
    "age", "pre_apt", "mono_total", "anc_total", "agitation",
    "post_gastric_tube", "crp_total"
]

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器",
    layout="wide"
)
st.title("急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器")
st.markdown("### 请填写以下信息，点击预测获取风险评估结果")

# ====================== 输入组件 ======================
bnp_total_num = st.number_input("基线BNP", min_value=0.0, value=0.0, step=1.0, format="%.2f")
hospital_day_num = st.number_input("住院天数", min_value=0.0, value=0.0, step=0.5, format="%.2f")
sbp_baseline_num = st.number_input("基线收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")
adl_total_num = st.number_input("基线自理能力评分", min_value=0.0, value=0.0, step=1.0, format="%.2f")
sbp_admit_num = st.number_input("入院收缩压", min_value=0.0, value=0.0, step=1.0, format="%.2f")
age_num = st.number_input("年龄", min_value=0.0, value=0.0, step=1.0, format="%.2f")
pre_apt = st.selectbox("术前是否使用抗凝抗板药物", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
mono_total_num = st.number_input("基线单核细胞计数", min_value=0.0, value=0.0, step=0.1, format="%.2f")
anc_total_num = st.number_input("基线中性粒细胞计数", min_value=0.0, value=0.0, step=0.1, format="%.2f")
agitation = st.selectbox(
    "术后躁动情况？",
    options=[0, 1, 2, 3],
    format_func=lambda x: {0: "无", 1: "轻度躁动", 2: "中度躁动", 3: "重度躁动"}[x]
)
post_gastric_tube = st.selectbox("术后是否留置胃管", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
crp_total_num = st.number_input("基线超敏C反应蛋白", min_value=0.0, value=0.0, step=0.1, format="%.2f")

# ====================== 预测 ======================
if st.button("预测"):
    # 按模型训练顺序构建特征数组
    feature_values = [
        bnp_total_num,
        hospital_day_num,
        sbp_baseline_num,
        adl_total_num,
        sbp_admit_num,
        age_num,
        pre_apt,
        mono_total_num,
        anc_total_num,
        agitation,
        post_gastric_tube,
        crp_total_num
    ]

    # 转换为DataFrame（带列名）
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # 获取高风险概率（二分类模型，概率数组[低风险概率, 高风险概率]）
    proba = model.predict_proba(input_df)[0]
    risk_prob = proba[1]  # 高风险概率

    # 根据自定义阈值划分风险等级（20%低风险阈值，80%高风险阈值）
    if risk_prob < 0.20:
        level = "低风险"
        advice = (
            f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于低风险。"
            "建议继续保持当前治疗方案，定期随访。"
        )
    elif risk_prob < 0.80:
        level = "中风险"
        advice = (
            f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于中风险。"
            "建议密切观察，遵医嘱进行相关检查。"
        )
    else:
        level = "高风险"
        advice = (
            f"模型预测您的症状性出血风险概率为 {risk_prob:.1%}，属于高风险。"
            "建议立即就医，加强监测和预防措施。"
        )

    # 显示预测结果
    st.subheader("📊 预测结果")
    st.write(f"**风险等级：{level}**")
    st.write(f"**高风险概率：{risk_prob:.2%}**")

    # 显示健康建议
    st.subheader("💡 健康建议")
    st.write(advice)

    # ====================== LIME解释 ======================
    st.subheader("🔍 LIME特征贡献解释")
    # 只取模型使用的12个特征列
    X_train_lime = test_dataset[feature_names].values
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=feature_names,
        class_names=['低风险', '高风险'],  # LIME仍用二分类名称
        mode='classification'
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.values.flatten(),
        predict_fn=model.predict_proba,
        num_features=12
    )
    lime_html = lime_exp.as_html(show_table=True)
    components.html(lime_html, height=600, scrolling=True)