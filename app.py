import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 🚨 เครื่องปั่นฟีเจอร์ (Interaction Features)
# ==========================================
def build_interaction_features(df):
    df_out = df.copy()
    
    if 'defendant_gender_group' in df_out.columns:
        is_female = (df_out['defendant_gender_group'] == 'female_only').astype(int)
    else:
        is_female = 0
        
    if 'is_violent' in df_out.columns:
        df_out['female_x_violent'] = is_female * df_out['is_violent']
    if 'is_high_value_theft' in df_out.columns and 'is_privileged_class' in df_out.columns:
        df_out['high_value_x_unprivileged'] = df_out['is_high_value_theft'] * (1 - df_out['is_privileged_class'])
    if 'is_privileged_class' in df_out.columns and 'crime_cat_deception' in df_out.columns:
        df_out['elite_x_deception'] = df_out['is_privileged_class'] * df_out['crime_cat_deception']
    elif 'is_privileged_class' in df_out.columns:
        df_out['elite_x_deception'] = 0
    if 'is_economic_crisis' in df_out.columns and 'is_survival_theft' in df_out.columns:
        df_out['crisis_x_survival'] = df_out['is_economic_crisis'] * df_out['is_survival_theft']
    if 'is_sympathetic_case' in df_out.columns:
        df_out['female_x_sympathetic'] = is_female * df_out['is_sympathetic_case']
        
    return df_out

# ==========================================
# 🎨 Page Configuration
# ==========================================
st.set_page_config(page_title="Historical AI Lab", page_icon="⚖️", layout="wide")

st.title("⚖️ Historical AI Lab: Decoding the Old Bailey")
st.markdown("""
**Statistical Disclaimer:** This AI model explores statistical probabilities based on historical patterns it learned, rather than stating absolute historical facts.
""")

# ==========================================
# 📦 1. Asset & Data Loading
# ==========================================
@st.cache_resource
def load_assets():
    xgb_path = 'Old_bailey_xgb_pipeline.pkl'
    rf_path = 'Old_bailey_rf_pipeline.pkl'
    encoder_path = 'label_encoder.pkl'
    x_test_path = 'X_test.csv'
    y_test_path = 'y_test.csv'
    
    try:
        xgb = joblib.load(xgb_path)
    except Exception as e:
        st.error(f"❌ โหลด XGBoost Pipeline พัง! \nสาเหตุ: {e}")
        xgb = None
        
    try:
        rf = joblib.load(rf_path)
    except Exception as e:
        st.warning(f"⚠️ โหลด Random Forest Pipeline ไม่ขึ้น! (ระบบจะใช้ XGBoost โชว์เดี่ยวแทน) \nสาเหตุ: {e}")
        rf = None 
        
    try:
        encoder = joblib.load(encoder_path)
        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path) 
    except Exception as e:
        st.error(f"❌ โหลดข้อมูล CSV พัง! สาเหตุ: {e}")
        return None, None, None, None, None
    
    native_cat_cols = [
        'defendant_gender_group', 'crime_location', 'season',
        'victim_scale', 'offence_complexity'
    ]
    for col in native_cat_cols:
        if col in x_test.columns:
            x_test[col] = x_test[col].astype('category')
            
    return xgb, rf, encoder, x_test, y_test

xgb_model, rf_model, label_encoder, X_test, y_test = load_assets()

if xgb_model is None:
    st.stop()

# ==========================================
# 📚 Mappings (เตรียมดิกชันนารี)
# ==========================================
crime_mapping = {
    'Any Charge': None,
    'Murder / Homicide 💀': 'crime_cat_kill',
    'General Theft 🥖': 'crime_cat_theft',
    'Violent Theft / Robbery 🔪': 'crime_cat_violentTheft',
    'Sexual Offense ⚠️': 'crime_cat_sexual',
    'Deception / Fraud 🤥': 'crime_cat_deception',
    'Breaking the Peace 👊': 'crime_cat_breakingPeace',
    'Royal Offenses / Treason 👑': 'crime_cat_royalOffences',
    'Property Damage 💥': 'crime_cat_damage'
}

era_mapping = {
    'is_early_bloody_code': '1. Early Bloody Code (Pre-1718)',
    'is_penal_crisis': '2. Penal Crisis (1776-1787)',
    'is_peelian_reform': '3. Peelian Reform (1820s)',
    'is_modern_legal_system': '4. Modern Victorian (Post-1850s)'
}

# กำหนดน้ำหนักการโหวตแบบไดนามิก ป้องกัน Class สลับที่
xgb_weight_dict = {'corporal': 0.6, 'death': 0.5, 'imprison': 0.5, 'miscPunish': 0.6, 'noPunish': 0.8, 'transport': 0.5}
rf_weight_dict  = {'corporal': 0.4, 'death': 0.5, 'imprison': 0.5, 'miscPunish': 0.4, 'noPunish': 0.2, 'transport': 0.5}

weight_xgb = np.array([xgb_weight_dict.get(cls, 0.5) for cls in label_encoder.classes_])
weight_rf  = np.array([rf_weight_dict.get(cls, 0.5) for cls in label_encoder.classes_])

# ==========================================
# ⚙️ 2. Sidebar: Targeted Case Randomizer
# ==========================================
st.sidebar.header("📜 1. Fetch Historical Case")

target_label = st.sidebar.selectbox("Filter by Primary Charge:", list(crime_mapping.keys()))
target_col = crime_mapping[target_label]

if st.sidebar.button("🎲 Draw Targeted Case File", use_container_width=True):
    if target_col:
        filtered_df = X_test[X_test[target_col] == 1]
        if not filtered_df.empty:
            chosen_idx = np.random.choice(filtered_df.index)
            st.session_state['random_idx'] = chosen_idx
        else:
            st.sidebar.error(f"No cases found for {target_label} in the test set.")
    else:
        st.session_state['random_idx'] = np.random.randint(0, len(X_test))

if 'random_idx' not in st.session_state:
    st.info("👈 Please select a charge and click 'Draw Targeted Case File' to begin.")
    st.stop()

idx = st.session_state['random_idx']
try:
    base_case = X_test.loc[[idx]].copy()
    actual_val = y_test.loc[idx].values[0]
except KeyError:
    idx = np.random.randint(0, len(X_test))
    base_case = X_test.iloc[[idx]].copy()
    actual_val = y_test.iloc[idx].values[0]

true_label = label_encoder.inverse_transform([actual_val])[0]

# ==========================================
# 🔎 3. Base Case Context (FULL DOSSIER)
# ==========================================
st.subheader("🔎 Official Case Dossier (Base Reality)")

committed_crime = "Unknown Offense"
reverse_mapping = {v: k for k, v in crime_mapping.items() if v is not None}
for col, readable_name in reverse_mapping.items():
    if col in base_case.columns and base_case[col].values[0] == 1:
        committed_crime = readable_name
        break

case_era = "Transitional / Unspecified Era"
for col, readable_era in era_mapping.items():
    if col in base_case.columns and base_case[col].values[0] == 1:
        case_era = readable_era
        break

st.markdown(f"### 🚨 Primary Charge: **{committed_crime}**")
st.markdown(f"### 🏛️ Legal Era: **{case_era}**")

loc = str(base_case['crime_location'].values[0])
season_val = str(base_case['season'].values[0])
gender = str(base_case['defendant_gender_group'].values[0])

d1, d2, d3 = st.columns(3)
with d1:
    st.markdown("#### 📍 Scene & Environment")
    st.write(f"**Location:** {loc}")
    st.write(f"**Season:** {season_val}")
    st.write(f"**Economic Crisis:** {'Yes' if base_case['is_economic_crisis'].values[0] == 1 else 'No'}")
    st.write(f"**Wartime:** {'Yes' if base_case['is_wartime'].values[0] == 1 else 'No'}")

with d2:
    st.markdown("#### 👤 Defendant Profile")
    st.write(f"**Gender:** {gender.replace('_', ' ').title()}")
    st.write(f"**Group Size:** {int(base_case['defendant_count'].values[0])}")
    st.write(f"**Privileged Class:** {'Yes' if base_case['is_privileged_class'].values[0] == 1 else 'No'}")
    st.write(f"**Sympathetic:** {'Yes' if base_case['is_sympathetic_case'].values[0] == 1 else 'No'}")

with d3:
    st.markdown("#### 🔪 Crime Details")
    st.write(f"**Violence:** {'Yes' if base_case['is_violent'].values[0] == 1 else 'No'}")
    st.write(f"**Victims:** {int(base_case['victim_count'].values[0])}")
    st.write(f"**Survival Theft:** {'Yes' if base_case['is_survival_theft'].values[0] == 1 else 'No'}")
    st.write(f"**High Value:** {'Yes' if base_case['is_high_value_theft'].values[0] == 1 else 'No'}")

st.divider()
st.markdown(f"### ⚖️ **Historical Verdict:** `{true_label}`")
st.divider()

# ==========================================
# 🧪 4. Sidebar: THE GOD-MODE WHAT-IF SETUP
# ==========================================
st.sidebar.divider()
st.sidebar.header("🧪 2. Re-shape the Reality")
st.sidebar.markdown("Modify *everything* including environment and scene.")

with st.sidebar.form("what_if_form"):
    st.markdown("#### 📍 1. Scene & Environment")
    loc_options = ['Unknown', 'Streets & Open Spaces', 'Private Residential', 'Hospitality & Leisure', 'Trade, Port & Industry', 'Institutional & State']
    n_loc = st.selectbox("Crime Location:", loc_options, index=loc_options.index(loc) if loc in loc_options else 0)
    
    season_options = ['Spring', 'Summer', 'Autumn', 'Winter']
    n_season = st.selectbox("Season:", season_options, index=season_options.index(season_val) if season_val in season_options else 0)
    
    n_eco = st.checkbox("During Economic Crisis", value=bool(base_case['is_economic_crisis'].values[0]))
    n_war = st.checkbox("During Wartime", value=bool(base_case['is_wartime'].values[0]))

    st.markdown("#### 👤 2. Defendant Profile")
    n_gender = st.selectbox("Gender Group:", ['male_only', 'female_only', 'mixed'], 
                            index=['male_only', 'female_only', 'mixed'].index(gender))
    n_def_cnt = st.slider("Number of Defendants:", 1.0, 3.0, float(base_case['defendant_count'].values[0]), 1.0)
    
    n_priv = st.checkbox("Privileged Class / Elite", value=bool(base_case['is_privileged_class'].values[0]), 
                         help="จำเลยเป็นคนรวย ขุนนาง หรือมีเส้นสายในสังคมหรือไม่?")
    n_symp = st.checkbox("Sympathetic Character", value=bool(base_case['is_sympathetic_case'].values[0]),
                         help="จำเลยดูน่าสงสาร เช่น เป็นเด็ก คนท้อง หรือขโมยอาหารเพราะหิวจัด?")

    st.markdown("#### 🔪 3. Crime Details")
    n_violent = st.checkbox("Used Violence", value=bool(base_case['is_violent'].values[0]))
    n_surv = st.checkbox("Theft for Survival", value=bool(base_case['is_survival_theft'].values[0]))
    n_high_val = st.checkbox("High Value Theft", value=bool(base_case['is_high_value_theft'].values[0]))
    n_vic_cnt = st.slider("Number of Victims:", 0.0, 3.0, float(base_case['victim_count'].values[0]), 1.0)
    
    st.markdown("#### ⚖️ 4. Crime Nature & Atmosphere")
    
    valid_charges = [k for k, v in crime_mapping.items() if v is not None]
    default_charge_idx = valid_charges.index(committed_crime) if committed_crime in valid_charges else 0
    n_charge = st.selectbox("Change Primary Charge:", valid_charges, index=default_charge_idx,
                            help="ลองเปลี่ยนข้อหาหลักดูว่าถ้ายุคสมัยเปลี่ยนไป ศาลจะตัดสินคดีนี้ต่างจากเดิมไหม?")
    
    n_gang = st.checkbox("Gang / Syndicate Involvement", value=bool(base_case['is_gang_crime'].values[0] if 'is_gang_crime' in base_case.columns else 0))
    n_petty = st.checkbox("Petty / Nuisance Offense", value=bool(base_case['is_nuisance_only'].values[0] if 'is_nuisance_only' in base_case.columns else 0))
    
    default_chaos = float(base_case['chaos_index'].values[0]) if 'chaos_index' in base_case.columns else 1.0
    
    n_chaos = st.slider("Social Chaos Index (1=Calm, 6=Riot):", 1.0, 6.0, default_chaos, 1.0,
                        help="ระดับความวุ่นวายในลอนดอนตอนนั้น (ยิ่งสูง ศาลอาจตัดสินโหดขึ้นเพื่อเชือดไก่ให้ลิงดู)")

    submit = st.form_submit_button("⏳ Run Time Machine", use_container_width=True)

# ==========================================
# ⏳ 5. TIME MACHINE Execution (ENSEMBLE)
# ==========================================
if submit:
    st.subheader("⏳ Time Machine: Ensemble Simulation Results")
    st.caption("🤖 Decision made by Supreme Court: XGBoost + Random Forest Ensemble (Class-Specific Weights)")
    
    historical_eras = {
        "1. Early Bloody Code\n(Pre-1718)": "is_early_bloody_code",
        "2. Penal Crisis\n(1776-1787)": "is_penal_crisis",
        "3. Peelian Reform\n(1820s)": "is_peelian_reform",
        "4. Modern Victorian\n(Post-1850s)": "is_modern_legal_system"
    }
    
    # ==========================================
    # 🛠️ [CLEAN CODE]: เตรียมข้อมูลพื้นฐานไว้ก่อนเข้าลูป
    # ==========================================
    base_modified = base_case.copy()
    
    # 1. ล้างข้อมูล 'ข้อหา' และ 'ยุค' ดั้งเดิมทิ้งทั้งหมด ป้องกัน Data Leakage
    all_dummy_charges = [c for c in crime_mapping.values() if c is not None]
    for c in all_dummy_charges:
        if c in base_modified.columns: 
            base_modified[c] = 0
            
    for e in historical_eras.values():
        if e in base_modified.columns: 
            base_modified[e] = 0
            
    # 2. ยัดพฤติการณ์พื้นฐาน (User Input) รวดเดียว
    behavior_updates = {
        'crime_location': n_loc, 'season': n_season,
        'is_economic_crisis': int(n_eco), 'is_wartime': int(n_war),
        'defendant_gender_group': n_gender, 'defendant_count': float(n_def_cnt),
        'victim_count': float(n_vic_cnt), 'is_privileged_class': int(n_priv),
        'is_sympathetic_case': int(n_symp), 'is_violent': int(n_violent),
        'is_survival_theft': int(n_surv), 'is_high_value_theft': int(n_high_val),
        'is_gang_crime': int(n_gang), 'is_nuisance_only': int(n_petty),
        'is_petty_solo': int(n_petty), 'chaos_index': float(n_chaos)
    }
    for col, val in behavior_updates.items():
        if col in base_modified.columns:
            base_modified[col] = val

    # 3. อัปเดตข้อหาใหม่ (ซิงค์ Dummy, Raw Category, และ Subcategory)
    new_charge_col = crime_mapping[n_charge]
    if new_charge_col is not None and new_charge_col in base_modified.columns:
        base_modified[new_charge_col] = 1
        
    if 'crime_category' in base_modified.columns and new_charge_col is not None:
        base_modified['crime_category'] = new_charge_col.replace('crime_cat_', '')
        
    if 'crime_subcategory' in base_modified.columns and new_charge_col is not None:
        if new_charge_col == 'crime_cat_kill': 
            base_modified['crime_subcategory'] = 'murder'
        elif new_charge_col == 'crime_cat_theft': 
            base_modified['crime_subcategory'] = 'simpleLarceny'
        else: 
            base_modified['crime_subcategory'] = 'other'

    # ==========================================
    # 🔄 เริ่มจำลองในแต่ละยุคสมัย
    # ==========================================
    era_columns = st.columns(4)
    era_year_mapping = {
        'is_early_bloody_code': 1700,
        'is_penal_crisis': 1780,
        'is_peelian_reform': 1825,
        'is_modern_legal_system': 1860
    }
    
    for i, (era_name, era_feature) in enumerate(historical_eras.items()):
        
        # ดึงคดีที่เตรียมไว้แล้วมาใช้เลย ไม่ต้องเซ็ตข้อมูลพฤติการณ์ใหม่ซ้ำซ้อน
        mod_case = base_modified.copy()
        
        # 4. อัปเดตยุคใหม่ให้ตรงกับ Loop ปัจจุบัน
        if era_feature in mod_case.columns:
            mod_case[era_feature] = 1
        
        if 'year' in mod_case.columns:
            mod_case['year'] = era_year_mapping.get(era_feature, 1800)
            
        # 5. รักษาสถานภาพ Category Type เพื่อไม่ให้ Pipeline Error
        cat_cols = ['defendant_gender_group', 'crime_location', 'season', 'victim_scale', 'offence_complexity']
        for c in cat_cols:
            if c in mod_case.columns:
                mod_case[c] = mod_case[c].astype('category')
        
        # ==========================================
        # ⚖️ การตัดสินโดยองค์คณะ 2 ผู้พิพากษา
        # ==========================================
        xgb_probs = xgb_model.predict_proba(mod_case)[0]
        
        rf_probs = None
        if rf_model is not None:
            try:
                rf_probs = rf_model.predict_proba(mod_case)[0]
            except Exception as e:
                # ปล่อยให้ rf_probs เป็น None อย่างถูกต้อง
                pass
                    
        # 🧠 รวมหัวกันเฉลี่ยคะแนนโดยใช้ Dynamic Weights (และ Fallback Math ที่ถูกหลักคณิตศาสตร์)
        if rf_probs is not None:
            ensemble_probs = (xgb_probs * weight_xgb) + (rf_probs * weight_rf)
            ensemble_probs = ensemble_probs / np.sum(ensemble_probs)
        else:
            # ถ้า RF พัง ให้ใช้ XGB เพียวๆ ไม่ต้องคูณน้ำหนักให้เพี้ยน
            ensemble_probs = xgb_probs
            
        pred_encoded = np.argmax(ensemble_probs)
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        confidence = ensemble_probs[pred_encoded] * 100
        
        # แง้มดูว่าศาลเสียงแตกไหม
        xgb_vote = label_encoder.inverse_transform([np.argmax(xgb_probs)])[0]
        rf_vote = label_encoder.inverse_transform([np.argmax(rf_probs)])[0] if rf_probs is not None else "N/A"
        
        # แสดงผลลัพธ์ลง Column UI
        with era_columns[i]:
            st.markdown(f"**{era_name}**")
            
            if xgb_vote == rf_vote:
                st.caption(f"🤝 ศาลเอกฉันท์: {xgb_vote.title()}")
            else:
                st.caption(f"⚡ ศาลเสียงแตก: XGB({xgb_vote.title()}) vs RF({rf_vote.title()})")
                
            # 🎨 แมปปิ้งสีให้ตรงกับความโหดของบทลงโทษ
            if pred_label == 'death': 
                st.error(f"💀 **{pred_label.upper()}**\n\n*(Conf: {confidence:.1f}%)*")
            elif pred_label == 'transport':
                st.warning(f"🚢 **{pred_label.upper()}**\n\n*(Conf: {confidence:.1f}%)*")
            elif pred_label == 'corporal': 
                st.warning(f"🩸 **{pred_label.upper()}**\n\n*(Conf: {confidence:.1f}%)*")
            elif pred_label == 'imprison': 
                st.info(f"⛓️ **{pred_label.upper()}**\n\n*(Conf: {confidence:.1f}%)*")
            else: 
                st.success(f"⚖️ **{pred_label.upper()}**\n\n*(Conf: {confidence:.1f}%)*")

    st.divider()