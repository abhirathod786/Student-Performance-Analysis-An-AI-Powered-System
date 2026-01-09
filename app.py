"""
STUDENT PERFORMANCE ANALYSIS- AN AI POWERED SYSTEM v4.0
Complete Edition with All Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="Student Performance Analysis- An AI Powered System", page_icon="🎓", layout="wide")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    
    .main-header {
        font-size: 3.5rem; font-weight: 900; text-align: center;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 30px; animation: fadeIn 1s;
    }
    @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
    @keyframes pulse { 0%, 100% {transform: scale(1);} 50% {transform: scale(1.05);} }
    
    .metric-box {
        background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        text-align: center; transition: all 0.3s;
        border-left: 5px solid #667eea;
    }
    .metric-box:hover { transform: translateY(-8px); box-shadow: 0 12px 30px rgba(0,0,0,0.15); }
    
    .metric-value {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: pulse 2s infinite;
    }
    
    .gradient-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px; border-radius: 20px; color: white;
        box-shadow: 0 15px 35px rgba(102,126,234,0.4);
        margin: 15px 0; transition: all 0.3s;
    }
    .gradient-card:hover { transform: translateY(-5px); }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 10px 30px rgba(255,107,107,0.4);
        animation: pulse 2s infinite; border-left: 6px solid #c92a2a;
    }
    
    .badge {
        display: inline-block; padding: 12px 25px; border-radius: 25px;
        font-weight: 700; margin: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    .badge-gold { background: linear-gradient(135deg, #ffd700, #ffed4e); color: #333; }
    .badge-silver { background: linear-gradient(135deg, #c0c0c0, #e8e8e8); color: #333; }
    
    .resource-card {
        background: white; padding: 20px; border-radius: 12px;
        margin: 12px 0; border-left: 4px solid #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: all 0.3s;
    }
    .resource-card:hover { transform: translateX(8px); border-left-width: 6px; }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; padding: 12px 35px; border-radius: 25px;
        font-weight: 700; transition: all 0.3s;
        box-shadow: 0 6px 18px rgba(102,126,234,0.4);
    }
    .stButton>button:hover { transform: translateY(-3px) scale(1.05); }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_resource
def load_system():
    models = {}
    try:
        with open('models/graduation_model.pkl', 'rb') as f:
            models['grad'] = pickle.load(f)
        with open('models/risk_model.pkl', 'rb') as f:
            models['risk'] = pickle.load(f)
        with open('models/le_graduation.pkl', 'rb') as f:
            models['le_grad'] = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            models['features'] = pickle.load(f)
        data = pd.read_csv('data/btech_ece_advanced.csv')
        
        # ADD THIS NEW CODE - Replace generic names with real Indian names
        indian_names = [
            'Rahul Sharma', 'Priya Patel', 'Arjun Kumar', 'Sneha Reddy', 'Vikram Singh',
            'Anjali Gupta', 'Rohan Mehta', 'Kavya Iyer', 'Aditya Joshi', 'Divya Nair',
            'Karthik Rao', 'Pooja Verma', 'Amit Shah', 'Riya Desai', 'Varun Pillai',
            'Neha Kulkarni', 'Siddharth Bhat', 'Ananya Menon', 'Nikhil Agarwal', 'Ishita Kapoor',
            'Harsh Pandey', 'Tanvi Shetty', 'Akash Malhotra', 'Shruti Nambiar', 'Manish Trivedi',
            'Deepika Bajaj', 'Rohit Chopra', 'Sakshi Ghosh', 'Vishal Yadav', 'Megha Bansal',
            'Gaurav Sinha', 'Nisha Khanna', 'Suresh Kumar', 'Pallavi Kaur', 'Rajesh Varma',
            'Swati Mishra', 'Ajay Tiwari', 'Preeti Saxena', 'Sandeep Rao', 'Kritika Sharma',
            'Abhishek Rathod', 'Sai Kiran', 'Gurugovind Patil', 'Aditi Bhosale', 'Chetan Gowda',
            'Shweta Hegde', 'Manoj Shetty', 'Vaishnavi Jain', 'Naveen Kumar', 'Rashmi Prabhu',
            'Prakash Naik', 'Lakshmi Reddy', 'Sanjay Hegde', 'Anusha Rao', 'Vinay Krishna',
            'Bhavana Shenoy', 'Sunil Patil', 'Rekha Bhat', 'Ramesh Pai', 'Sowmya Kulkarni',
            'Ashish Nayak', 'Vidya Desai', 'Ravi Shankar', 'Pavitra Gowda', 'Mahesh Rao',
            'Shilpa Hegde', 'Yogesh Shetty', 'Varsha Prabhu', 'Girish Kumar', 'Manju Kamath'
        ]
        
        # Extend list if needed
        while len(indian_names) < len(data):
            indian_names.extend([f'Student {i+1}' for i in range(len(data) - len(indian_names))])
        
        # Replace names in data
        data['name'] = indian_names[:len(data)]
        # END OF NEW CODE
        
        return models, data
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
models, data = load_system()

def predict(student, models):
    X = np.array([student[f] if f in student.index else 0 for f in models['features']]).reshape(1, -1)
    g_pred = models['grad'].predict(X)[0]
    g_prob = models['grad'].predict_proba(X)[0]
    risk = models['risk'].predict(X)[0]
    return {'grad': models['le_grad'].inverse_transform([g_pred])[0],
            'grad_conf': g_prob[g_pred] * 100, 'risk': risk}

# Generate Progress Data
def gen_progress(student):
    cgpa = student.get('overall_cgpa', 0)
    # Semesters with 6-month gaps: Sem1, Sem2, Sem3, Sem4
    semesters = ['Sem 1\n(Jan-Jun)', 'Sem 2\n(Jul-Dec)', 'Sem 3\n(Jan-Jun)', 'Sem 4\n(Jul-Dec)']
    
    # Generate realistic progression
    if cgpa >= 7.5:
        cgpa_hist = [cgpa - 0.8, cgpa - 0.5, cgpa - 0.3, cgpa]
    elif cgpa >= 6.0:
        cgpa_hist = [cgpa - 0.6, cgpa - 0.4, cgpa - 0.2, cgpa]
    else:
        cgpa_hist = [cgpa - 0.4, cgpa - 0.3, cgpa - 0.15, cgpa]
    
    # Attendance progression
    att_base = student.get('overall_attendance', 0)
    att_hist = [max(60, att_base - 15), max(65, att_base - 10), max(70, att_base - 5), att_base]
    
    return pd.DataFrame({'Semester': semesters, 'CGPA': cgpa_hist, 'Attendance': att_hist})
# Peer Comparison
def peer_compare(student, data):
    cgpa_pct = (data['overall_cgpa'] < student.get('overall_cgpa', 0)).sum() / len(data) * 100
    att_pct = (data['overall_attendance'] < student.get('overall_attendance', 0)).sum() / len(data) * 100
    code_pct = (data['coding_test_score'] < student.get('coding_test_score', 0)).sum() / len(data) * 100
    return {
        'rank': int((100 - cgpa_pct) * len(data) / 100),
        'cgpa_pct': cgpa_pct, 'att_pct': att_pct, 'code_pct': code_pct,
        'total': len(data)
    }

# Resources
def get_resources(student):
    resources = []
    cgpa = student.get('overall_cgpa', 0)
    coding = student.get('coding_test_score', 0)
    
    if cgpa < 6.5:
        resources.extend([
            {'icon': '📚', 'name': 'NPTEL', 'desc': 'IIT video lectures', 'link': 'nptel.ac.in', 'priority': 'HIGH'},
            {'icon': '📖', 'name': 'Khan Academy', 'desc': 'Math & Science basics', 'link': 'khanacademy.org', 'priority': 'HIGH'},
        ])
    
    if coding < 70:
        resources.extend([
            {'icon': '💻', 'name': 'LeetCode', 'desc': 'Coding practice', 'link': 'leetcode.com', 'priority': 'HIGH'},
            {'icon': '🚀', 'name': 'HackerRank', 'desc': 'Programming challenges', 'link': 'hackerrank.com', 'priority': 'HIGH'},
            {'icon': '📚', 'name': 'GeeksforGeeks', 'desc': 'DSA tutorials', 'link': 'geeksforgeeks.org', 'priority': 'HIGH'},
        ])
    
    resources.extend([
        {'icon': '🎓', 'name': 'Coursera', 'desc': 'Professional courses', 'link': 'coursera.org', 'priority': 'MED'},
        {'icon': '📝', 'name': 'edX', 'desc': 'University courses', 'link': 'edx.org', 'priority': 'MED'},
    ])
    return resources

# Achievements
def get_achievements(student):
    ach = []
    cgpa = student.get('overall_cgpa', 0)
    att = student.get('overall_attendance', 0)
    backs = student.get('current_backlogs', 0)
    code = student.get('coding_test_score', 0)
    
    if cgpa >= 9.0:
        ach.append({'icon': '🏆', 'title': 'Outstanding Scholar', 'class': 'badge-gold'})
    elif cgpa >= 8.0:
        ach.append({'icon': '⭐', 'title': 'Excellent Student', 'class': 'badge-silver'})
    
    if att >= 95:
        ach.append({'icon': '📅', 'title': 'Perfect Attendance', 'class': 'badge-gold'})
    if backs == 0:
        ach.append({'icon': '🎯', 'title': 'Zero Backlogs', 'class': 'badge-gold'})
    if code >= 85:
        ach.append({'icon': '💻', 'title': 'Coding Master', 'class': 'badge-gold'})
    
    return ach

# Email Generator
def gen_email(student, pred):
    return f"""
Subject: 🚨 Academic Alert - {student['name']}

Dear {student['name']},

CURRENT STATUS:
• CGPA: {student['overall_cgpa']:.2f}
• Risk Score: {pred['risk']:.1f}/100
• Attendance: {student.get('overall_attendance', 0):.1f}%
• Backlogs: {int(student.get('current_backlogs', 0))}

ACTION REQUIRED:
{'🚨 CRITICAL: Immediate intervention needed!' if pred['risk'] > 70 else '⚠️ WARNING: Action recommended'}

NEXT STEPS:
1. Meet academic advisor within 48 hours
2. Attend all classes without exception
3. Check detailed plan in system

Support: advisor@university.edu

Best regards,
Student Performance System
"""

# Report Generator
def gen_report(student, pred, prog_df, ach, peer):
    return f"""
╔═══════════════════════════════════════════════════════════╗
║      COMPREHENSIVE STUDENT PERFORMANCE REPORT             ║
╚═══════════════════════════════════════════════════════════╝

STUDENT: {student['name']} ({student['student_id']})
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 ACADEMIC PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CGPA: {student['overall_cgpa']:.2f}
Attendance: {student.get('overall_attendance', 0):.1f}%
Backlogs: {int(student.get('current_backlogs', 0))}
Coding Score: {student.get('coding_test_score', 0):.0f}/100

🎯 AI PREDICTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Graduation: {pred['grad']} ({pred['grad_conf']:.1f}% confidence)
Risk Score: {pred['risk']:.1f}/100
Status: {'CRITICAL' if pred['risk'] > 70 else 'HIGH' if pred['risk'] > 50 else 'MODERATE'}

📈 PROGRESS (Last 6 Months)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{prog_df.to_string(index=False)}

🏆 ACHIEVEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join([f"{a['icon']} {a['title']}" for a in ach]) if ach else 'No achievements yet'}

👥 PEER COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank: {peer['rank']}/{peer['total']}
CGPA Percentile: {peer['cgpa_pct']:.1f}%
Attendance Percentile: {peer['att_pct']:.1f}%

🎯 RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{f"⚠️ CRITICAL: Immediate intervention required!" if pred['risk'] > 70 else f"Focus on improving CGPA to 7.0+" if student['overall_cgpa'] < 6.5 else "Maintain excellence!"}

═══════════════════════════════════════════════════════════
Generated by Student Performance Analysis- an AI Powered System 
"""

# Sidebar
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px;'>
    <div style='font-size: 4rem; animation: pulse 2s infinite;'>🎓</div>
    <h2 style='background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900;'>Ultimate System</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard", "🔍 Student Analysis", "📈 Progress Tracking",
    "👥 Peer Comparison", "📚 Resources", "🏆 Achievements",
    "📧 Email Alerts", "📄 Export Report",
], label_visibility="collapsed")

at_risk = (data['risk_score'] > 50).sum()
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class='gradient-card' style='padding: 15px;'>
    <p>👥 Students: <b>{len(data)}</b></p>
    <p>⚠️ At Risk: <b>{at_risk}</b></p>
    <p>✅ Status: <b>🟢 Online</b></p>
</div>
""", unsafe_allow_html=True)

# DASHBOARD
if page == "🏠 Dashboard":
    st.markdown("<div class='main-header'>🎓 Student Performance Analysis - An AI Powered System 🎓</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>AI-Powered • Real-Time • Personalized</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    excellent = (data['overall_cgpa'] >= 8.0).sum()
    
    with col1:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-value'>{len(data)}</div>
            <div style='color: #666; font-weight: 600;'>📚 Students</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""<div class='metric-box' style='border-left-color: #ff6b6b;'>
            <div class='metric-value' style='background: linear-gradient(135deg, #ff6b6b, #ee5a6f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{at_risk}</div>
            <div style='color: #666; font-weight: 600;'>⚠️ At Risk</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class='metric-box' style='border-left-color: #51cf66;'>
            <div class='metric-value' style='background: linear-gradient(135deg, #51cf66, #37b24d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{excellent}</div>
            <div style='color: #666; font-weight: 600;'>⭐ Excellent</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-value'>{data['overall_cgpa'].mean():.2f}</div>
            <div style='color: #666; font-weight: 600;'>📊 Avg CGPA</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Distribution")
        risk_cats = pd.cut(data['risk_score'], bins=[0,30,50,70,100], labels=['Low','Medium','High','Critical'])
        counts = risk_cats.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
            marker=dict(color=['#51cf66','#ffd93d','#ff9966','#ff6b6b']),
            text=counts.values, textposition='auto'))
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 CGPA Distribution")
        fig = go.Figure(go.Histogram(x=data['overall_cgpa'], nbinsx=25,
            marker=dict(color=data['overall_cgpa'], colorscale='RdYlGn')))
        fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# STUDENT ANALYSIS
elif page == "🔍 Student Analysis":
    st.markdown("# 🔍 Student Deep Analysis")
    student_id = st.selectbox("Select Student", data['student_id'].tolist())
    
    if student_id:
        student = data[data['student_id'] == student_id].iloc[0]
        pred = predict(student, models)
        
        st.markdown(f"""<div class='gradient-card'>
            <h1>{student['name']}</h1>
            <p><b>ID:</b> {student['student_id']} | <b>Gender:</b> {student['gender']}</p>
            <h2 style='margin-top: 20px;'>CGPA: {student['overall_cgpa']:.2f}</h2>
        </div>""", unsafe_allow_html=True)
        
        if pred['risk'] > 70:
            st.markdown(f"""<div class='alert-critical'>
                <h2>🚨 CRITICAL RISK</h2>
                <p style='font-size: 1.2rem;'>Risk Score: {pred['risk']:.1f}/100</p>
            </div>""", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📅 Attendance", f"{student.get('overall_attendance', 0):.1f}%")
        with col2:
            st.metric("📚 Backlogs", int(student.get('current_backlogs', 0)))
        with col3:
            st.metric("💻 Coding", f"{student.get('coding_test_score', 0):.0f}/100")
        with col4:
            st.metric("💼 Internships", int(student.get('internships_completed', 0)))

# PROGRESS TRACKING
elif page == "📈 Progress Tracking":
    st.markdown("# 📈 Progress Tracking")
    student_id = st.selectbox("Select Student", data['student_id'].tolist())
    
    if student_id:
        student = data[data['student_id'] == student_id].iloc[0]
        prog = gen_progress(student)
        
        st.markdown(f"## 📊 {student['name']}'s Progress Over Time")
        
        st.markdown("### 📊 CGPA Progress by Semester")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prog['Semester'], 
            y=prog['CGPA'], 
            mode='lines+markers', 
            name='CGPA',
            line=dict(color='#667eea', width=4),
            marker=dict(size=12, color='#667eea', line=dict(color='white', width=2))
        ))
        fig.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 10], title='CGPA'),
            xaxis=dict(title='Semester (6-month periods)'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📅 Attendance Progress by Semester")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=prog['Semester'], 
            y=prog['Attendance'], 
            mode='lines+markers', 
            name='Attendance',
            line=dict(color='#51cf66', width=4),
            marker=dict(size=12, color='#51cf66', line=dict(color='white', width=2))
        ))
        fig2.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 100], title='Attendance %'),
            xaxis=dict(title='Semester (6-month periods)'),
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.dataframe(prog, use_container_width=True)

# PEER COMPARISON
elif page == "👥 Peer Comparison":
    st.markdown("# 👥 Peer Comparison Analysis")
    student_id = st.selectbox("Select Student", data['student_id'].tolist())
    
    if student_id:
        student = data[data['student_id'] == student_id].iloc[0]
        peer = peer_compare(student, data)
        
        st.markdown(f"## 📊 {student['name']}'s Standing")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-value'>{peer['rank']}</div>
                <div style='color: #666;'>Rank out of {peer['total']}</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-value'>{peer['cgpa_pct']:.1f}%</div>
                <div style='color: #666;'>CGPA Percentile</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("### 📊 Percentile Comparison")
        fig = go.Figure(go.Bar(
            x=['CGPA', 'Attendance', 'Coding'],
            y=[peer['cgpa_pct'], peer['att_pct'], peer['code_pct']],
            marker=dict(color=['#667eea', '#51cf66', '#ffd93d']),
            text=[f"{peer['cgpa_pct']:.1f}%", f"{peer['att_pct']:.1f}%", f"{peer['code_pct']:.1f}%"],
            textposition='auto'
        ))
        fig.update_layout(height=400, yaxis=dict(range=[0,100], title='Percentile'))
        st.plotly_chart(fig, use_container_width=True)

# RESOURCES
elif page == "📚 Resources":
    st.markdown("# 📚 Personalized Resource Library")
    student_id = st.selectbox("Select Student", data['student_id'].tolist())
    
    if student_id:
        student = data[data['student_id'] == student_id].iloc[0]
        resources = get_resources(student)
        
        st.markdown(f"## 🎯 Recommended for {student['name']}")
        
        for res in resources:
            priority_color = {'HIGH': '#ff6b6b', 'MED': '#ffd93d'}.get(res['priority'], '#667eea')
            st.markdown(f"""<div class='resource-card'>
                <h3>{res['icon']} {res['name']}</h3>
                <p>{res['desc']}</p>
                <p><b>Link:</b> <a href='https://{res["link"]}' target='_blank'>{res['link']}</a></p>
                <span style='background: {priority_color}; color: white; padding: 5px 15px; border-radius: 15px; font-weight: 700;'>
                    {res['priority']} PRIORITY
                </span>
            </div>""", unsafe_allow_html=True)

# ACHIEVEMENTS
elif page == "🏆 Achievements":
    st.markdown("# 🏆 Student Achievements & Badges")
    student_id = st.selectbox("Select Student", data['student_id'].tolist())
    
    if student_id:
        student = data[data['student_id'] == student_id].iloc[0]
        achievements = get_achievements(student)
        
        st.markdown(f"## 🌟 {student['name']}'s Achievements")
        
        if achievements:
            for ach in achievements:
                st.markdown(f"""<span class='badge {ach["class"]}'>{ach['icon']} {ach['title']}</span>""", unsafe_allow_html=True)
        else:
            st.info("No achievements yet. Keep working hard! 💪")

# EMAIL ALERTS
elif page == "📧 Email Alerts":
    st.markdown("# 📧 Email Alert System")
    st.markdown("Generate email notifications for at-risk students")
    
    risk_threshold = st.slider("Risk Threshold", 0, 100, 50)
    at_risk_students = data[data['risk_score'] > risk_threshold]
    
    st.markdown(f"## ⚠️ {len(at_risk_students)} Students Need Alerts")
    
    for _, student in at_risk_students.head(5).iterrows():
        pred = predict(student, models)
        email_content = gen_email(student, pred)
        
        with st.expander(f"📧 {student['name']} - Risk: {pred['risk']:.1f}"):
            st.code(email_content, language='text')
            st.button(f"📨 Send Email to {student['name']}", key=student['student_id'])

# EXPORT REPORT
elif page == "📄 Export Report":
    st.markdown("# 📄 Export Detailed Reports")
    student_id = st.selectbox("Select Student", data['student_id'].tolist())
    
    if student_id:
        student = data[data['student_id'] == student_id].iloc[0]
        pred = predict(student, models)
        prog = gen_progress(student)
        ach = get_achievements(student)
        peer = peer_compare(student, data)
        
        report = gen_report(student, pred, prog, ach, peer)
        
        st.markdown(f"## 📊 Report for {student['name']}")
        st.text_area("Report Preview", report, height=400)
        
        st.download_button(
            label="📥 Download Report (.txt)",
            data=report,
            file_name=f"Report_{student['student_id']}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(data))
    with col2:
        st.metric("At Risk", at_risk)
    with col3:
        st.metric("Avg CGPA", f"{data['overall_cgpa'].mean():.2f}")
    


