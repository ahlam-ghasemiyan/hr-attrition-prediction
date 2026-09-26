
# Hr_dp.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix 
import joblib
import numpy as np
from HR_data import prepare_data

# اجرای تابع آماده‌سازی داده (که فایل‌های pkl را هم می‌سازد)
print("--- در حال آماده‌سازی داده‌ها ---")
X_train_scaled, X_test_scaled, y_train, y_test, input_dim = prepare_data()

# تبدیل به آرایه Numpy و نوع float32 (برای جلوگیری از خطای تنسورفلو)
X_train_scaled = np.array(X_train_scaled, dtype=np.float32)
X_test_scaled = np.array(X_test_scaled, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

L2_REG = 0.001 

# --- ساخت مدل ---
model = Sequential([
    # اصلاح مهم: استفاده از لایه Input صریح برای رفع خطای batch_shape
    Input(shape=(input_dim,)), 
    
    # لایه ۱
    Dense(256, activation='relu', kernel_regularizer=l2(L2_REG)),
    Dropout(0.4),

    # لایه ۲
    Dense(128, activation='relu', kernel_regularizer=l2(L2_REG)),
    Dropout(0.3),

    # لایه ۳
    Dense(64, activation='relu'),

    # لایه ۴
    Dense(32, activation='relu'),

    # لایه خروجی
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=20, 
    restore_best_weights=True 
)

# آموزش مدل
print("\n--- شروع آموزش مدل Deep Learning ---")
history = model.fit(
     X_train_scaled,
     y_train,
     epochs=100, # کاهش تعداد ایپاک برای تست سریع‌تر (می‌توانید زیاد کنید)
     batch_size=32,
     validation_data=(X_test_scaled, y_test),
     callbacks=[early_stopping], 
     verbose=1
)
print("--- آموزش کامل شد ---")

# ارزیابی
y_pred_proba = model.predict(X_test_scaled).flatten()

THRESHOLD = 0.4
y_pred = (y_pred_proba > THRESHOLD).astype(int)

auc_score = roc_auc_score(y_test, y_pred_proba)

print("\n\n################ گزارش عملکرد ################")
print(f"**AUC Score: {auc_score:.4f}**\n")
print(classification_report(y_test, y_pred))

# ذخیره مدل
model_name = 'professional_attrition_dl_model.h5'
model.save(model_name)
print(f"\n✅ مدل نهایی با نام {model_name} ذخیره شد.")