# #HR_data.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import joblib

# def prepare_data(file_path='WA_Fn-UseC_-HR-Employee-Attrition.csv', test_size=0.2, random_state=42):
#     """
#     داده‌ها را بارگذاری، پیش‌پردازش و مقیاس‌گذاری می‌کند.
#     متغیرهای آماده شده برای مدل‌سازی را برمی‌گرداند.
#     """
#     df = pd.read_csv(file_path)

#     cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
#     df = df.drop(cols_to_drop, axis=1)

#     df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
#     df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#     X = df_encoded.drop('Attrition', axis=1)
#     y = df_encoded['Attrition']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

#     scaler = StandardScaler()

#     numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()

#     numerical_cols = [col for col in numerical_cols if not col.startswith(('BusinessTravel_', 'Department_', 'EducationField_', 'Gender_', 'JobRole_', 'MaritalStatus_', 'OverTime_'))]

#     X_train_scaled = X_train.copy()
#     X_test_scaled = X_test.copy()

#     X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
#     X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
#     joblib.dump(scaler, 'scaler.pkl')
#     joblib.dump(X.columns, 'model_columns.pkl')
#     joblib.dump(numerical_cols, 'numerical_cols.pkl') 

#     print("--- داده‌ها با موفقیت آماده شدند و scaler و model_columns ذخیره شدند. ---")
    
#     return X_train_scaled, X_test_scaled, y_train, y_test, X.shape[1]
# if __name__ == '__main__':
#     X_train_scaled, X_test_scaled, y_train, y_test, input_dim = prepare_data()
#     print(f"شکل مجموعه آموزش: {X_train_scaled.shape}")



# HR_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

def prepare_data(file_path='WA_Fn-UseC_-HR-Employee-Attrition.csv', test_size=0.2, random_state=42):
    """
    داده‌ها را بارگذاری، پیش‌پردازش و مقیاس‌گذاری می‌کند.
    فایل‌های کمکی (scaler, columns) را برای API ذخیره می‌کند.
    """
    # بررسی وجود فایل
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"فایل دیتاست پیدا نشد: {file_path}")

    df = pd.read_csv(file_path)

    # حذف ستون‌های بی‌استفاده
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

    # تبدیل ستون هدف به عدد
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # شناسایی ستون‌های دسته‌بندی (متنی)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # تبدیل به One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # جدا کردن ویژگی‌ها و هدف
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']

    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # --- اصلاح مهم: انتخاب دقیق ستون‌های عددی برای اسکیل کردن ---
    # فقط ستون‌هایی که واقعا عدد پیوسته هستند (نه ۰ و ۱ های حاصل از get_dummies)
    # ما لیست ستون‌های اصلی عددی را قبل از اینکدینگ استخراج می‌کنیم یا با شرط بررسی می‌کنیم
    all_num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    # حذف ستون‌های باینری (که فقط ۰ و ۱ دارند) از لیست اسکیلینگ
    numerical_cols = [col for col in all_num_cols if X_train[col].nunique() > 2]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # فقط ستون‌های عددی واقعی را اسکیل می‌کنیم
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # --- ذخیره‌سازی برای استفاده در API ---
    # نکته مهم: حتما تبدیل به list شود تا json serializable باشد
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'model_columns.pkl')     # اصلاح: ذخیره به عنوان لیست ساده
    joblib.dump(numerical_cols, 'numerical_cols.pkl') 

    print("✅ داده‌ها آماده شدند.")
    print("✅ فایل‌های scaler.pkl, model_columns.pkl, numerical_cols.pkl ذخیره شدند.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.shape[1]

if __name__ == '__main__':
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, input_dim = prepare_data()
        print(f"شکل مجموعه آموزش: {X_train_scaled.shape}")
    except Exception as e:
        print(f"خطا: {e}")