

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
from tensorflow.keras.models import load_model
import shap 
import matplotlib.pyplot as plt

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'show_charts' not in st.session_state:
    st.session_state.show_charts = False
if 'shap_results' not in st.session_state:
    st.session_state.shap_results = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

def toggle_charts():
    """ تابع برای تغییر وضعیت نمایش نمودارها """
    st.session_state.show_charts = not st.session_state.show_charts


st.set_page_config(page_title="HR Attrition DL Predictor", layout="wide")



@st.cache_data(show_spinner=False)
def get_eda_data(file_path='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """ بارگذاری داده‌های خام برای نمودارهای EDA (MonthlyIncome اضافه شد)"""
    try:
        df = pd.read_csv(file_path)
        df_eda = df[['Department', 'Attrition', 'MaritalStatus', 'MonthlyIncome']].copy()
        df_eda['Attrition_Numeric'] = df_eda['Attrition'].map({'Yes': 1, 'No': 0})
        return df_eda
    except FileNotFoundError:
        st.error(f"فایل دیتاست '{file_path}' پیدا نشد. نمودارهای تحلیلی اجرا نخواهند شد.")
        return None

try:
    model = load_model('professional_attrition_dl_model.h5')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    numerical_cols_to_scale = joblib.load('numerical_cols.pkl') 
    
    df_eda = get_eda_data() 

except Exception as e:
    st.error(f"خطا در بارگذاری فایل‌های مدل یا دیتاست: {e}")
    st.stop()



def set_background(image_file):
    """ (بخش CSS بدون تغییر) """
    try:
        with open(image_file, "rb") as f:
            img_bytes = f.read()
        encoded_img = base64.b64encode(img_bytes).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_img}");
                background-size: cover;
                background-attachment: fixed;
            }}
            
            h1, h2, h3, h4 {{ color: white !important; text-shadow: 2px 2px 4px #000000; }}
            label, .st-emotion-cache-6v09g0, .st-emotion-cache-1215pkj {{ color: white !important; text-shadow: 1px 1px 2px #000000; }}

            .stSlider > div > div > div:nth-child(2) {{ background-color: default !important; }}
            .stSlider > div > div > div:nth-child(2) > div {{ background-color: default !important; }}
            
            div.stButton {{ display: flex; justify-content: center; width: 100%; margin-top: 30px; margin-bottom: 20px; }}
            div.stButton > button {{
                background-color: white !important; color: #04366e !important; border: 2px solid #04366e !important;
                padding: 15px 60px !important; font-size: 20px !important; border-radius: 12px !important;
                font-weight: bold; transition: all 0.2s; min-width: 300px;
            }}
            div.stButton > button:hover {{ background-color: #f0f2f6 !important; }}
            
            .stApp > header, .css-vk3gh2, .css-1y4p85n, .css-1dp54x9, .css-1r6dm1s {{
                background-color: rgba(255, 255, 255, 0.9); padding: 10px; border-radius: 10px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("تصویر پس‌زمینه 'hr.jpg' پیدا نشد. برنامه بدون پس‌زمینه اجرا می‌شود.")

@st.cache_resource(show_spinner=False)
def get_explainer(_model, _model_columns):
    """ ساخت DeepExplainer با استفاده از cache_resource """
    try:
        background_data = np.load('X_train_sample_scaled.npy')
    except FileNotFoundError:
        background_data = np.zeros((10, len(_model_columns))) 
    
    return shap.DeepExplainer(_model, background_data)

def plot_attrition_by_department_chart(df_eda):
    if df_eda is None: return None
        
    attrition_rate = df_eda.groupby('Department')['Attrition_Numeric'].mean().reset_index()
    attrition_rate['Attrition_Percent'] = attrition_rate['Attrition_Numeric'] * 100
    
    fig, ax = plt.subplots(figsize=(6, 4)) 
    attrition_rate = attrition_rate.sort_values(by='Attrition_Percent', ascending=False)
    
    ax.bar(attrition_rate['Department'], attrition_rate['Attrition_Percent'], color='#1f77b4')
    ax.set_title('Attrition Rate by Department | نرخ ترک خدمت بر اساس دپارتمان', fontsize=10, color='black')
    ax.set_ylabel('Attrition Rate (%)', fontsize=8, color='black')
    
    plt.xticks(rotation=45, ha='right', fontsize=7, color='black')
    plt.yticks(fontsize=7, color='black')
    
    plt.tight_layout()
    fig.patch.set_alpha(0.9) 
    ax.patch.set_alpha(0.9)
    
    return fig

def plot_attrition_by_marital_status_chart(df_eda):
    """ محاسبه و رسم نرخ ترک خدمت بر اساس وضعیت تأهل (با سایز کوچک) """
    if df_eda is None: return None
        
    attrition_rate = df_eda.groupby('MaritalStatus')['Attrition_Numeric'].mean().reset_index()
    attrition_rate['Attrition_Percent'] = attrition_rate['Attrition_Numeric'] * 100
    
    fig, ax = plt.subplots(figsize=(6, 4)) 
    attrition_rate = attrition_rate.sort_values(by='Attrition_Percent', ascending=False)
    
    colors = {'Single': '#ff7f0e', 'Married': '#2ca02c', 'Divorced': '#d62728'}
    bar_colors = [colors[status] for status in attrition_rate['MaritalStatus']]
    
    ax.bar(attrition_rate['MaritalStatus'], attrition_rate['Attrition_Percent'], color=bar_colors)
    ax.set_title('Attrition Rate by Marital Status | نرخ ترک خدمت بر اساس تاهل', fontsize=10, color='black')
    ax.set_ylabel('Attrition Rate (%)', fontsize=8, color='black')
    
    plt.xticks(rotation=0, fontsize=7, color='black')
    plt.yticks(fontsize=7, color='black')
    
    plt.tight_layout()
    fig.patch.set_alpha(0.9) 
    ax.patch.set_alpha(0.9)
    
    return fig
    
# ⚠️ تابع جدید رسم نمودار توزیع درآمد
def plot_attrition_by_income_distribution_chart(df_eda):
    """ رسم توزیع درآمد ماهانه برای افراد باقی‌مانده و ترک کرده """
    if df_eda is None: return None
        
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # تفکیک داده‌ها بر اساس Attrition
    income_yes = df_eda[df_eda['Attrition'] == 'Yes']['MonthlyIncome']
    income_no = df_eda[df_eda['Attrition'] == 'No']['MonthlyIncome']
    
    ax.hist(income_no, bins=30, alpha=0.6, label='Stayed (مانده)', color='#52c41a') # سبز
    ax.hist(income_yes, bins=30, alpha=0.6, label='Left (ترک کرده)', color='#ff4d4f') # قرمز
    
    ax.set_title('Monthly Income Distribution vs Attrition | توزیع درآمد در مقابل ترک خدمت', fontsize=10, color='black')
    ax.set_xlabel('Monthly Income (درآمد ماهانه)', fontsize=8, color='black')
    ax.set_ylabel('Number of Employees (تعداد کارمندان)', fontsize=8, color='black')
    ax.legend(fontsize=7)
    
    plt.tight_layout()
    fig.patch.set_alpha(0.9) 
    ax.patch.set_alpha(0.9)
    
    return fig



set_background('hr.jpg') 
st.title("💼 **Employee Attrition Prediction | پیش‌بینی ترک خدمت کارکنان**")
st.markdown("---")



options_travel = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
options_education = {1: '1 - Below College', 2: '2 - College', 3: '3 - Bachelor', 4: '4 - Master', 5: '5 - Doctor'}

st.markdown('### 📝 **Enter Employee Data | اطلاعات کارمند را وارد کنید**')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Personal & Core Data | مشخصات اصلی")
    Age = st.slider("Age | سن", 18, 60, 30, key='Age')
    Gender = st.selectbox("Gender | جنسیت", ['Male', 'Female'], key='Gender')
    MaritalStatus = st.selectbox("Marital Status | وضعیت تأهل", ['Single', 'Married', 'Divorced'], key='MaritalStatus')
    DistanceFromHome = st.slider("Distance From Home (Miles) | فاصله از خانه (مایل)", 1, 29, 5, key='DistanceFromHome')
    BusinessTravel = st.selectbox("Business Travel | سفر کاری", options_travel, key='BusinessTravel')
    OverTime = st.selectbox("OverTime | اضافه‌کاری", ['Yes', 'No'], key='OverTime')
    st.markdown("---")
    JobInvolvement = st.slider("Job Involvement | تعهد شغلی (1:Low, 4:High)", 1, 4, 3, key='JobInvolvement')
    EnvironmentSatisfaction = st.slider("Environment Satisfaction | رضایت محیطی (1:Low, 4:High)", 1, 4, 3, key='EnvironmentSatisfaction')


with col2:
    st.markdown("#### Job & Salary Data | شغل و حقوق")
    Department = st.selectbox("Department | دپارتمان", ['Research & Development', 'Sales', 'Human Resources'], key='Department')
    JobRole = st.selectbox("Job Role | نقش شغلی", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], key='JobRole')
    JobLevel = st.slider("Job Level | سطح شغلی", 1, 5, 2, key='JobLevel')
    MonthlyIncome = st.number_input("Monthly Income | درآمد ماهانه", 1000, 20000, 5000, step=100, key='MonthlyIncome')
    DailyRate = st.number_input("Daily Rate | نرخ روزانه", 102, 1499, 1000, key='DailyRate')
    HourlyRate = st.number_input("Hourly Rate | نرخ ساعتی", 30, 100, 65, key='HourlyRate')
    MonthlyRate = st.number_input("Monthly Rate | نرخ ماهانه", 2094, 26999, 15000, key='MonthlyRate')
    PercentSalaryHike = st.slider("Percent Salary Hike | افزایش حقوق (%)", 11, 25, 14, key='PercentSalaryHike')
    PerformanceRating = st.selectbox("Performance Rating | رتبه عملکرد", [3, 4], index=0, key='PerformanceRating')
    st.markdown("---")
    JobSatisfaction = st.slider("Job Satisfaction | رضایت شغلی (1:Low, 4:High)", 1, 4, 3, key='JobSatisfaction')
    RelationshipSatisfaction = st.slider("Relationship Satisfaction | رضایت روابط (1:Low, 4:High)", 1, 4, 3, key='RelationshipSatisfaction')


with col3:
    st.markdown("#### Experience & Education | سابقه و تحصیلات")
    TotalWorkingYears = st.slider("Total Working Years | کل سال‌های کار", 0, 40, 8, key='TotalWorkingYears')
    YearsAtCompany = st.slider("Years At Company | سال‌های حضور در شرکت", 0, 40, 5, key='YearsAtCompany')
    YearsInCurrentRole = st.slider("Years In Current Role | سال‌ها در نقش فعلی", 0, 18, 2, key='YearsInCurrentRole')
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion | سال از آخرین ترفیع", 0, 15, 1, key='YearsSinceLastPromotion')
    YearsWithCurrManager = st.slider("Years With Curr Manager | سال‌ها با مدیر فعلی", 0, 17, 2, key='YearsWithCurrManager')
    NumCompaniesWorked = st.slider("Num Companies Worked | تعداد شرکت‌های قبلی", 0, 9, 2, key='NumCompaniesWorked')
    TrainingTimesLastYear = st.slider("Training Times Last Year | دفعات آموزش در سال", 0, 6, 3, key='TrainingTimesLastYear')
    WorkLifeBalance = st.slider("Work Life Balance | تعادل کار و زندگی (1:Poor, 4:Best)", 1, 4, 3, key='WorkLifeBalance')
    StockOptionLevel = st.slider("Stock Option Level | سطح سهام اختیار", 0, 3, 1, key='StockOptionLevel')
    st.markdown("---")
    Education = st.select_slider("Education | تحصیلات", options=list(options_education.keys()), format_func=lambda x: options_education[x], key='Education')
    EducationField = st.selectbox("Education Field | رشته تحصیلی", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], key='EducationField')



if st.button("🔮 Predict Attrition Risk | پیش‌بینی ریسک ترک خدمت"):
    
    with st.spinner('Processing Data and Running Deep Learning Model... | درحال اجرای مدل یادگیری عمیق...'):
        
        data = {
            'Age': Age, 'Attrition': 'No', 'BusinessTravel': BusinessTravel, 'DailyRate': DailyRate, 'Department': Department, 
            'DistanceFromHome': DistanceFromHome, 'Education': Education, 'EducationField': EducationField, 'EmployeeCount': 1, 
            'EmployeeNumber': 1, 'EnvironmentSatisfaction': EnvironmentSatisfaction, 'Gender': Gender, 'HourlyRate': HourlyRate, 
            'JobInvolvement': JobInvolvement, 'JobLevel': JobLevel, 'JobRole': JobRole, 'JobSatisfaction': JobSatisfaction, 
            'MaritalStatus': MaritalStatus, 'MonthlyIncome': MonthlyIncome, 'MonthlyRate': MonthlyRate, 'NumCompaniesWorked': NumCompaniesWorked, 
            'Over18': 'Y', 'OverTime': OverTime, 'PercentSalaryHike': PercentSalaryHike, 'PerformanceRating': PerformanceRating, 
            'RelationshipSatisfaction': RelationshipSatisfaction, 'StandardHours': 80, 'StockOptionLevel': StockOptionLevel, 
            'TotalWorkingYears': TotalWorkingYears, 'TrainingTimesLastYear': TrainingTimesLastYear, 'WorkLifeBalance': WorkLifeBalance, 
            'YearsAtCompany': YearsAtCompany, 'YearsInCurrentRole': YearsInCurrentRole, 'YearsSinceLastPromotion': YearsSinceLastPromotion, 
            'YearsWithCurrManager': YearsWithCurrManager
        }
        
        user_df = pd.DataFrame(data, index=[0])
        
        cols_to_drop_input = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber', 'Attrition']
        user_df = user_df.drop(cols_to_drop_input, axis=1)

        user_df_encoded = pd.get_dummies(user_df, drop_first=True)
        missing_cols = set(model_columns) - set(user_df_encoded.columns)
        for c in missing_cols:
            user_df_encoded[c] = 0
        user_data_final = user_df_encoded[model_columns].iloc[[0]]
        
        features_to_scale = user_data_final[numerical_cols_to_scale]
        user_data_final[numerical_cols_to_scale] = scaler.transform(features_to_scale)
        
        prediction_proba = model.predict(user_data_final)[0][0]
        prediction_percent = prediction_proba * 100
        
        st.markdown("---")
        st.subheader("📊 Deep Learning Prediction Results | نتایج پیش‌بینی")
        
        st.metric(label="Attrition Probability | احتمال ترک خدمت", value=f"{prediction_percent:.2f}%")

        if prediction_percent >= 50:
            st.error(f"🔴 **High Risk | ریسک بالا:** Probability is **{prediction_percent:.2f}%**. **Immediate action** recommended to retain the employee. ")
            st.balloons() 
        elif prediction_percent >= 30:
            st.warning(f"🟡 **Medium Risk | ریسک متوسط:** Probability is **{prediction_percent:.2f}%**. Needs **further review and retention effort**.")
        else:
            st.success(f"🟢 **Low Risk | ریسک پایین:** Probability is only **{prediction_percent:.2f}%**. Low immediate concern.")
        
        st.markdown("<br>", unsafe_allow_html=True) 
        if prediction_percent >= 50:
            summary_message = f"❌ **این فرد احتمالاً شرکت را ترک خواهد کرد.** (بر اساس احتمال ترک خدمت {prediction_percent:.2f}%)"
            color_style = "background-color: #ffebe8; border-left: 5px solid #ff4d4f; color: #cf1322;"
        else:
            summary_message = f"✅ **این فرد احتمالاً در شرکت باقی خواهد ماند.** (بر اساس احتمال ترک خدمت {prediction_percent:.2f}%)"
            color_style = "background-color: #f6ffed; border-left: 5px solid #52c41a; color: #237804;"
            
        st.markdown(
            f'<div style="{color_style} padding: 15px; border-radius: 8px; text-align: center;">'
            f'<p style="margin: 0; font-size: 1.2em; font-weight: bold;">{summary_message}</p></div>', 
            unsafe_allow_html=True
        )

        try:
            explainer = get_explainer(model, model_columns)
            shap_values = explainer.shap_values(user_data_final.values)

            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values_class_1 = shap_values[1][0]
            else:
                shap_values_class_1 = shap_values[0][0]

            st.session_state.shap_results = shap_values_class_1
            st.session_state.feature_names = user_data_final.columns
            st.session_state.prediction_made = True
            st.session_state.show_charts = False # بستن نمودار بعد از هر پیش‌بینی جدید
            
        except Exception as e:
            st.error(f"خطا در اجرای تحلیل SHAP: {e}")
            st.session_state.prediction_made = False 



if st.session_state.prediction_made:
    st.markdown("---")
    
    col_l, col_c, col_r = st.columns([1, 1, 1])
    with col_c:
        chart_button_text = "▼ پنهان کردن نمودارها" if st.session_state.show_charts else "نمودارهای تحلیلی ▲"
        st.button(chart_button_text, on_click=toggle_charts, key='toggle_btn')
    
    if st.session_state.show_charts:
        st.markdown("---")
        
        st.subheader("📊 **Contextual & Organizational Analysis | تحلیل زمینه‌ای و سازمانی**")
        eda_col1, eda_col2, eda_col3 = st.columns(3) 
        
        with eda_col1:
            st.markdown("#### Attrition Rate by Department | نرخ ترک خدمت بر اساس دپارتمان")
            eda_fig = plot_attrition_by_department_chart(df_eda)
            if eda_fig:
                st.pyplot(eda_fig)
            else:
                st.warning("⚠️ داده‌های اصلی (WA_Fn-UseC_-HR-Employee-Attrition.csv) برای رسم این نمودار پیدا نشد.")

        with eda_col2:
            st.markdown("#### Attrition Rate by Marital Status | نرخ ترک خدمت بر اساس تاهل")
            eda_fig_marital = plot_attrition_by_marital_status_chart(df_eda)
            if eda_fig_marital:
                st.pyplot(eda_fig_marital)
            else:
                st.warning("⚠️ داده‌های اصلی (WA_Fn-UseC_-HR-Employee-Attrition.csv) برای رسم این نمودار پیدا نشد.")
                
        with eda_col3: 
            st.markdown("#### Monthly Income Distribution | توزیع درآمد ماهانه")
            eda_fig_income = plot_attrition_by_income_distribution_chart(df_eda)
            if eda_fig_income:
                st.pyplot(eda_fig_income)
            else:
                st.warning("⚠️ داده‌های اصلی (WA_Fn-UseC_-HR-Employee-Attrition.csv) برای رسم این نمودار پیدا نشد.")


st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: This prediction is based on the Deep Learning model trained on historical HR data. | سلب مسئولیت: این پیش‌بینی بر اساس مدل Deep Learning آموزش دیده بر روی داده‌های تاریخی HR است.")