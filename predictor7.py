# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# 加载模型和数据
model = joblib.load('xgboost.pkl')
test_dataset = pd.read_csv('test_dataset.csv')  # 如果编码不是 utf-8，可添加 encoding='gbk'

# 模型使用的12个特征（必须与训练时的顺序一致）
feature_names = [
    "bnp_total", "hospital_day", "sbp_baseline", "adl_total", "sbp_admit",
    "age", "pre_apt", "mono_total", "anc_total", "agitation",
    "post_gastric_tube", "crp_total"
]

# ====================== 页面配置 ======================
st.set_page_config(page_title="急性缺血性脑卒中血管内治疗术后症状性出血转化风险预测器", layout="wide")
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

    # 调试输出（可删除）
    st.write("特征数量:", len(feature_values))
    st.write("特征值:", feature_values)

    # 转换为 DataFrame（带列名）
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # 模型预测
    predicted_class = model.predict(input_df)[0]  # 0：低风险，1：高风险
    predicted_proba = model.predict_proba(input_df)[0]

    # 显示预测结果
    st.subheader("📊 预测结果")
    risk_label = "高风险" if predicted_class == 1 else "低风险"
    st.write(f"**风险等级：{risk_label}**")
    st.write(f"**风险概率：** 低风险 {predicted_proba[0]:.2%} | 高风险 {predicted_proba[1]:.2%}")

    # 健康建议
    st.subheader("💡 健康建议")
    prob = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"模型预测您的症状性出血风险为高风险（概率{prob:.1f}%）。"
            "建议密切监测神经系统症状，遵医嘱进行影像学复查，控制血压、血糖，预防并发症。"
        )
    else:
        advice = (
            f"模型预测您的症状性出血转化风险为低风险（概率{prob:.1f}%）。"
            "建议继续保持当前治疗方案，定期随访，如有不适及时就医。"
        )
    st.write(advice)

    # ====================== LIME解释 ======================
    st.subheader("🔍 LIME特征贡献解释")
    # 只取模型使用的12个特征列
    X_train_lime = test_dataset[feature_names].values
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=feature_names,
        class_names=['低风险', '高风险'],
        mode='classification'
    )
    # 注意：这里用 input_df.values.flatten() 或 input_df.iloc[0].values
    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.values.flatten(),
        predict_fn=model.predict_proba,
        num_features=12
    )
    lime_html = lime_exp.as_html(show_table=True)
    components.html(lime_html, height=600, scrolling=True)