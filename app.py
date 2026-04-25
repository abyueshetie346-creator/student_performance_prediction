import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu




# የ GitHub icon ብቻ ለመደበቅ የሚረዳ ኮድ
hide_github_only = """
    <style>
    /* የ GitHub iconን ብቻ ለመደበቅ */
    .viewerBadge_container__1QS13, .viewerBadge_link__1QS13 {
        display: none !important;
    }
    #GithubIcon {
        visibility: hidden;
    }
    /* በስተቀኝ በኩል ያለውን የ GitHub ሊንክ ለመደበቅ */
    button[title="View source on GitHub"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_github_only, unsafe_allow_html=True)
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide"
)

# --- LANGUAGE SESSION STATE ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

def set_lang(lang_code):
    st.session_state.lang = lang_code

# --- TRANSLATION DICTIONARY ---
texts = {
    'EN': {
        'nav_home': "Home", 'nav_predict': "AI Predictor", 'nav_analytics': "Analytics", 'nav_info': "Info",
        'main_title': "EduPredict AI: Student Success Portal",
        'main_subtitle': "Predict student outcomes using Artificial Intelligence.",
        # Home Content
        'home_intro': "Welcome to the future of education management. Our AI analyzes historical and behavioral data to ensure no student is left behind.",
        'how_to_use': "How to Use:",
        'step_1': "1. Enter student data in the sidebar.",
        'step_2': "2. Go to 'AI Predictor' to see the results.",
        'step_3': "3. Check 'Analytics' to see the driving factors.",
        'system_importance': "Why is this important?",
        'importance_text': "Early identification of 'at-risk' students allows for timely intervention, helping to reduce dropout rates and improve graduation success.",
        # Sidebar
        'sidebar_title': "🛠️ Input Data", 'g1': "G1 Grade", 'g2': "G2 Grade", 
        'absences': "Absences", 'failures': "Past Failures", 'study': "Study Time",
        'sex': "Sex", 'age': "Age", 'female': "Female", 'male': "Male",
        'study_levels': ["Low", "Mid", "High", "Elite"],
        # Prediction Page
        'classes': ["At Risk", "Moderate", "High Achiever"],
        'conf': "Confidence", 'pred_title': "Final Prediction", 'radar_title': "Performance Radar",
        'desc_at_risk': "⚠️ **Urgent:** This student shows high risk of failing. Recommend immediate counseling and academic support.",
        'desc_moderate': "🟡 **Observation:** The student is performing at an average level but could improve with more focused study time.",
        'desc_high': "🌟 **Excellent:** This student is performing exceptionally well. Potential candidate for leadership roles or scholarships.",
        # Rewards & Celebration
        'reward_high': "🏆 **Reward:** Academic Excellence Badge awarded!",
        'reward_mod': "⭐ **Reward:** Progress Badge! You are on the right track.",
        'excellent_msg': "EXCELLENT WORK! 👏",
        # Analytics Page
        'analysis_title': "Driving Factors Behind Prediction",
        'analysis_desc': "Feature Importance shows which factors (like grades or absences) most influenced the AI's result for this specific student profile.",
        'advice_header': "📖 Advice for Students",
        'advice_1': "Focus on G1 foundations.",
        'advice_2': "Maintain >90% attendance.",
        'advice_3': "Consistent study habits."
    },
    'AM': {
        'nav_home': "መነሻ", 'nav_predict': "ትንበያ", 'nav_analytics': "ትንታኔ", 'nav_info': "መረጃ",
        'main_title': "EduPredict AI: የተማሪዎች ውጤት መተንበያ",
        'main_subtitle': "ሰው ሰራሽ አስተውሎትን በመጠቀም የተማሪዎችን ውጤት መተንበያ",
        # Home Content
        'home_intro': "እንኳን ወደ የወደፊት የትምህርት አመራር በደህና መጡ። የእኛ AI ተማሪዎችን ለመርዳት ታሪካዊ እና ባህሪ መረጃዎችን ይተነትናል።",
        'how_to_use': "እንዴት መጠቀም ይቻላል:",
        'step_1': "1. በጎን በኩል ባለው ሳጥን ውስጥ የተማሪውን መረጃ ያስገቡ።",
        'step_2': "2. ውጤቱን ለማየት ወደ 'ትንበያ' ገጽ ይሂዱ።",
        'step_3': "3. ዋና ዋና ምክንያቶችን ለማየት ወደ 'ትንታኔ' ገጽ ይሂዱ።",
        'system_importance': "ይህ ለምን አስፈላጊ ሆነ?",
        'importance_text': "አስጊ ሁኔታ ላይ ያሉ ተማሪዎችን ቀድሞ መለየት አስፈላጊውን ድጋፍ በጊዜ ለመስጠት እና የተማሪዎችን ማቋረጥ ለመቀነስ ይረዳል።",
        # Sidebar
        'sidebar_title': "🛠️ መረጃ ያስገቡ", 'g1': "የG1 ውጤት", 'g2': "የG2 ውጤት", 
        'absences': "የቀሩባቸው ቀናት", 'failures': "የወደቁባቸው ክፍሎች", 'study': "የጥናት ትጋት",
        'sex': "ጾታ", 'age': "ዕድሜ", 'female': "ሴት", 'male': "ወንድ",
        'study_levels': ["ዝቅተኛ", "መካከለኛ", "ከፍተኛ", "በጣም ከፍተኛ"],
        # Prediction Page
        'classes': ["አስጊ (አደጋ ላይ)", "መካከለኛ", "በጣም ጎበዝ"],
        'conf': "እርግጠኝነት", 'pred_title': "የመጨረሻ ትንበያ", 'radar_title': "የውጤት ራዳር",
        'desc_at_risk': "⚠️ **አስቸኳይ:** ይህ ተማሪ የመውደቅ ዕድሉ ከፍተኛ ነው። በአስቸኳይ የምክር አገልግሎት እና የትምህርት ድጋፍ ያስፈልገዋል።",
        'desc_moderate': "🟡 **ምልከታ:** ተማሪው በመካከለኛ ደረጃ ላይ ይገኛል ነገር ግን በጥናት ሰዓቱ ላይ ትኩረት ቢያደርግ ሊሻሻል ይችላል።",
        'desc_high': "🌟 **በጣም ጎበዝ:** ይህ ተማሪ በጥሩ ውጤት ላይ ይገኛል። ለሽልማት ወይም ለልዩ ድጋፍ እጩ ሊሆን ይችላል።",
        # Rewards & Celebration
        'reward_high': "🏆 **ሽልማት:** የላቀ የትምህርት ውጤት ሜዳሊያ ተሰጥቷል!",
        'reward_mod': "⭐ **ሽልማት:** የታታሪነት ባጅ! በጥሩ ጎዳና ላይ ነህ።",
        'excellent_msg': "በጣም ድንቅ ስራ! 👏",
        # Analytics Page
        'analysis_title': "ለትንበያው መነሻ የሆኑ ዋና ምክንያቶች",
        'analysis_desc': "ይህ ትንታኔ የትኛው መረጃ (ለምሳሌ ውጤት ወይም መቅረት) በ AI ውሳኔ ላይ ከፍተኛ ተጽዕኖ እንዳሳደረ ያሳያል።",
        'advice_header': "📖 ለተማሪዎች ምክሮች",
        'advice_1': "በመጀመሪያው ክፍለ ጊዜ (G1) ጠንክሮ መስራት።",
        'advice_2': "ከክፍል አለመቅረት (>90%)።",
        'advice_3': "የአጠናን ልምድን ማዳበር።"
    }
}

L = texts[st.session_state.lang]

# --- PREMIUM CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #000000 100%); color: #ffffff; }
    .glass-card { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(15px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.15); padding: 25px; margin-bottom: 20px; }
    .main-title { background: linear-gradient(to right, #00d2ff, #91eae4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 45px; font-weight: 800; text-align: center; }
    h3 { color: #00d2ff; }
    .reward-box { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-top: 10px; border: 1px solid rgba(255, 255, 255, 0.3); }
    </style>
    """, unsafe_allow_html=True)

# --- TOP LANGUAGE BUTTONS ---
col_empty, col_btn = st.columns([8, 2])
with col_btn:
    btn_en, btn_am = st.columns(2)
    if btn_en.button("🇺🇸 EN"): set_lang('EN')
    if btn_am.button("🇪🇹 AM"): set_lang('AM')

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('models/scaler.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        features = joblib.load('models/features.pkl')
        return scaler, rf_model, features
    except: return None, None, None

scaler, rf_model, features = load_assets()

# --- TOP NAVIGATION ---
selected = option_menu(
    menu_title=None, 
    options=[L['nav_home'], L['nav_predict'], L['nav_analytics'], L['nav_info']], 
    icons=["house", "robot", "graph-up", "info-circle"], 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "rgba(255,255,255,0.1)"},
        "nav-link-selected": {"background-color": "#00d2ff", "color": "black", "font-weight": "bold"},
    }
)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"### {L['sidebar_title']}")
    g1 = st.slider(L['g1'], 0, 20, 12)
    g2 = st.slider(L['g2'], 0, 20, 11)
    absences = st.number_input(L['absences'], 0, 50, 4)
    failures = st.selectbox(L['failures'], [0, 1, 2, 3])
    studytime = st.select_slider(L['study'], options=[1, 2, 3, 4], format_func=lambda x: L['study_levels'][x-1])
    st.markdown("---")
    sex = st.radio(L['sex'], [L['female'], L['male']], horizontal=True)
    age = st.number_input(L['age'], 15, 30, 18)

# PREDICTION LOGIC
if scaler:
    input_dict = {'age': age, 'G1': g1, 'G2': g2, 'failures': failures, 'studytime': studytime, 'absences': absences, 
                  'sex': 1 if L['male'] in sex else 0, 'internet': 1, 'activities': 1}
    input_df = pd.DataFrame([input_dict])[features]
    prediction = rf_model.predict(scaler.transform(input_df))[0]
    prob = rf_model.predict_proba(scaler.transform(input_df))[0]

# --- PAGE 1: HOME ---
if selected == L['nav_home']:
    st.markdown(f'<p class="main-title">{L["main_title"]}</p>', unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{L['main_subtitle']}</h3>", unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><p style="font-size: 18px; text-align: center;">{L["home_intro"]}</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="glass-card"><h3>{L["how_to_use"]}</h3><p>{L["step_1"]}</p><p>{L["step_2"]}</p><p>{L["step_3"]}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="glass-card"><h3>{L["system_importance"]}</h3><p>{L["importance_text"]}</p></div>', unsafe_allow_html=True)

# --- PAGE 2: AI PREDICTOR ---
elif selected == L['nav_predict']:
    st.title(f"🤖 {L['nav_predict']}")
    col_l, col_r = st.columns([1, 1.2])
    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        colors = ["#FF4B2B", "#FFB75E", "#00B09B"]
        st.subheader(L['pred_title'])
        st.markdown(f"<h1 style='color: {colors[prediction]};'>{L['classes'][prediction]}</h1>", unsafe_allow_html=True)
        
        # Dynamic Description & Rewards
        if prediction == 0: 
            st.error(L['desc_at_risk'])
        elif prediction == 1: 
            st.warning(L['desc_moderate'])
            st.markdown(f'<div class="reward-box" style="background-color: rgba(255, 183, 94, 0.2);">{L["reward_mod"]}</div>', unsafe_allow_html=True)
        else: 
            st.success(L['desc_high'])
            st.balloons()
            
            # --- FIXED SOUND LOGIC ---
            # Using a reliable "Tada/Success" sound link
            sound_url = "https://actions.google.com/sounds/v1/human_voices/applause_clapping.ogg"
            st.markdown(f'<iframe src="{sound_url}" allow="autoplay" style="display:none"></iframe>', unsafe_allow_html=True)
            
            st.markdown(f"<h2 style='text-align:center; color:#FFD700;'>{L['excellent_msg']}</h2>", unsafe_allow_html=True)
            st.markdown(f'<div class="reward-box" style="background-color: rgba(0, 176, 155, 0.2);">{L["reward_high"]}</div>', unsafe_allow_html=True)
        
        fig_g = go.Figure(go.Indicator(mode = "gauge+number", value = max(prob) * 100, gauge = {'bar': {'color': colors[prediction]}}, title = {'text': f"{L['conf']} %"}))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(L['radar_title'])
        fig_r = go.Figure(data=go.Scatterpolar(r=[g1, g2, max(0, 20-absences/2), studytime*5], theta=['G1', 'G2', 'Attendance', 'Study'], fill='toself', line_color='#00d2ff'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 20])), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: ANALYTICS ---
elif selected == L['nav_analytics']:
    st.title(f"📊 {L['nav_analytics']}")
    st.markdown(f'<div class="glass-card"><h3>{L["analysis_title"]}</h3><p>{L["analysis_desc"]}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#00d2ff'])
    fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 4: INFO ---
elif selected == L['nav_info']:
    st.title("ℹ️ Resources / መመሪያዎች")
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown(f'<div class="glass-card"><h3>{L["advice_header"]}</h3><ul><li>{L["advice_1"]}</li><li>{L["advice_2"]}</li><li>{L["advice_3"]}</li></ul></div>', unsafe_allow_html=True)
    with c_b:
        st.markdown('<div class="glass-card"><h3>🤖 About the AI / ስለ ሲስተሙ</h3><p>This tool uses a Random Forest model trained on academic data.</p><p><b>Developed by Group 3</b></p></div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(f"<br><hr><center>EduPredict AI | Developed by Computer Science Students | © 2026</center>", unsafe_allow_html=True)