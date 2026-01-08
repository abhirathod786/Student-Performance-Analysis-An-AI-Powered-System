"""
PHASE 1: ADVANCED B.TECH ECE DATASET GENERATOR
50+ Features | Time-Series Data | Multi-Target Predictions
"""

import pandas as pd
import numpy as np
import os

def generate_advanced_btech_dataset(n_students=300):
    """
    Generate comprehensive B.Tech ECE dataset with 50+ features
    """
    
    np.random.seed(42)
    
    print("="*70)
    print(" "*10 + "ADVANCED B.TECH ECE DATASET GENERATION")
    print("="*70)
    
    # ==========================================
    # BASIC INFORMATION
    # ==========================================
    
    student_ids = [f"ECE2022{str(i+1).zfill(3)}" for i in range(n_students)]
    names = [f"Student_{i+1}" for i in range(n_students)]
    genders = np.random.choice(['Male', 'Female'], n_students, p=[0.70, 0.30])
    
    # ==========================================
    # SEMESTER-WISE PERFORMANCE (Time-Series)
    # ==========================================
    
    print("\n📊 Generating semester-wise performance data...")
    
    # Base CGPA trajectory for each student
    base_cgpa = np.random.normal(7.0, 1.5, n_students)
    base_cgpa = np.clip(base_cgpa, 4.5, 9.5)
    
    # Generate realistic semester progression
    sem_cgpas = {}
    sem_attendance = {}
    sem_backlogs = {}
    
    for sem in range(1, 9):
        # CGPA progression with natural variation
        if sem == 1:
            sem_cgpa = base_cgpa + np.random.normal(0.2, 0.3, n_students)
        else:
            # Students improve or decline based on previous performance
            prev_cgpa = sem_cgpas[f'sem{sem-1}_cgpa']
            improvement = np.random.normal(0, 0.3, n_students)
            # Good students tend to maintain, struggling students vary more
            improvement = np.where(prev_cgpa > 7.5, improvement * 0.5, improvement)
            sem_cgpa = prev_cgpa + improvement
        
        # Semester difficulty adjustment
        difficulty = {1: 0.3, 2: 0.2, 3: 0, 4: -0.2, 5: -0.4, 6: -0.2, 7: 0.1, 8: 0.2}
        sem_cgpa = sem_cgpa + difficulty.get(sem, 0)
        sem_cgpa = np.clip(sem_cgpa, 4.0, 10.0)
        
        sem_cgpas[f'sem{sem}_cgpa'] = sem_cgpa.round(2)
        
        # Attendance (correlated with CGPA)
        attendance = sem_cgpa * 9 + np.random.normal(10, 8, n_students)
        attendance = np.clip(attendance, 45, 100)
        sem_attendance[f'sem{sem}_attendance'] = attendance.round(1)
        
        # Backlogs per semester
        backlog_prob = np.where(sem_cgpa < 5.5, 0.6, np.where(sem_cgpa < 6.5, 0.3, 0.05))
        has_backlog = np.random.random(n_students) < backlog_prob
        backlogs = np.where(has_backlog, np.random.choice([1, 2, 3], n_students, p=[0.6, 0.3, 0.1]), 0)
        sem_backlogs[f'sem{sem}_backlogs'] = backlogs
    
    # Overall metrics
    overall_cgpa = sum([sem_cgpas[f'sem{i}_cgpa'] for i in range(1, 9)]) / 8
    overall_cgpa = overall_cgpa.round(2)
    
    overall_attendance = sum([sem_attendance[f'sem{i}_attendance'] for i in range(1, 9)]) / 8
    overall_attendance = overall_attendance.round(1)
    
    total_backlogs = sum([sem_backlogs[f'sem{i}_backlogs'] for i in range(1, 9)])
    current_backlogs = sem_backlogs['sem8_backlogs'] + np.random.poisson(0.5, n_students)
    current_backlogs = np.clip(current_backlogs, 0, 8)
    
    # ==========================================
    # ACADEMIC ENGAGEMENT (Week-by-Week)
    # ==========================================
    
    print("📚 Generating academic engagement features...")
    
    # Assignment submissions (per semester average)
    assignment_rate = overall_cgpa * 9.5 + np.random.normal(5, 12, n_students)
    assignment_rate = np.clip(assignment_rate, 30, 100).round(1)
    
    # On-time submission rate
    ontime_rate = assignment_rate * 0.8 + np.random.normal(0, 10, n_students)
    ontime_rate = np.clip(ontime_rate, 20, 100).round(1)
    
    # Late submissions
    late_submissions = ((100 - ontime_rate) / 100 * 10).round(0).astype(int)
    
    # Quiz performance
    quiz_avg = overall_cgpa * 9 + np.random.normal(5, 10, n_students)
    quiz_avg = np.clip(quiz_avg, 30, 100).round(1)
    
    # Lab performance
    lab_performance = overall_cgpa * 9.5 + np.random.normal(0, 8, n_students)
    lab_performance = np.clip(lab_performance, 40, 100).round(1)
    
    # Lab attendance (usually higher than theory)
    lab_attendance = overall_attendance + np.random.normal(5, 5, n_students)
    lab_attendance = np.clip(lab_attendance, 50, 100).round(1)
    
    # Project scores
    project_score = overall_cgpa * 9 + np.random.normal(5, 10, n_students)
    project_score = np.clip(project_score, 40, 100).round(1)
    
    # Class participation (1-10)
    participation = overall_cgpa * 1.2 + np.random.normal(0, 1.5, n_students)
    participation = np.clip(participation, 2, 10).round(1)
    
    # ==========================================
    # DIGITAL ENGAGEMENT (LMS/Online)
    # ==========================================
    
    print("💻 Generating digital engagement metrics...")
    
    # LMS login frequency (per week)
    lms_logins = overall_cgpa * 2 + np.random.normal(8, 5, n_students)
    lms_logins = np.clip(lms_logins, 2, 30).round(0).astype(int)
    
    # Time spent on LMS (hours per week)
    lms_time = overall_cgpa * 1.5 + np.random.normal(5, 3, n_students)
    lms_time = np.clip(lms_time, 1, 25).round(1)
    
    # Video lecture completion rate
    video_completion = overall_cgpa * 9 + np.random.normal(10, 15, n_students)
    video_completion = np.clip(video_completion, 20, 100).round(1)
    
    # Discussion forum posts
    forum_posts = np.random.poisson(overall_cgpa * 0.5, n_students).astype(int)
    forum_posts = np.clip(forum_posts, 0, 20)
    
    # Resource downloads
    resource_downloads = np.random.poisson(overall_cgpa * 1.5, n_students).astype(int)
    resource_downloads = np.clip(resource_downloads, 5, 50)
    
    # ==========================================
    # STUDY PATTERNS
    # ==========================================
    
    print("📖 Generating study pattern data...")
    
    # Study hours per week
    study_hours = overall_cgpa * 3 + np.random.normal(10, 8, n_students)
    study_hours = np.clip(study_hours, 5, 50).round(1)
    
    # Library visits per week
    library_visits = overall_cgpa * 0.5 + np.random.normal(2, 2, n_students)
    library_visits = np.clip(library_visits, 0, 10).round(1)
    
    # Study group participation
    study_group = np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often'], 
                                   n_students, p=[0.2, 0.3, 0.35, 0.15])
    
    # Peak study time
    study_time = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 
                                  n_students, p=[0.15, 0.20, 0.35, 0.30])
    
    # ==========================================
    # EXTRACURRICULAR & TECHNICAL ACTIVITIES
    # ==========================================
    
    print("🏆 Generating extracurricular activities...")
    
    # Internships
    internship_prob = np.where(overall_cgpa > 7.5, 0.7, np.where(overall_cgpa > 6.5, 0.4, 0.15))
    internships = np.random.binomial(3, internship_prob).astype(int)
    
    # Internship ratings (1-5) if completed
    internship_rating = np.where(internships > 0, 
                                 overall_cgpa * 0.5 + np.random.normal(1, 0.5, n_students),
                                 0)
    internship_rating = np.clip(internship_rating, 0, 5).round(1)
    
    # Certifications
    cert_prob = overall_cgpa * 0.08
    certifications = np.random.binomial(8, cert_prob).astype(int)
    
    # Technical papers presented
    papers = np.random.choice([0, 1, 2, 3, 4], n_students, p=[0.5, 0.25, 0.15, 0.07, 0.03])
    
    # Hackathons participated
    hackathons = np.random.choice([0, 1, 2, 3, 4, 5], n_students, p=[0.4, 0.25, 0.2, 0.1, 0.04, 0.01])
    
    # Competitions won
    competitions_won = np.where(hackathons > 0, 
                                np.random.binomial(hackathons, 0.3),
                                0).astype(int)
    
    # Open source contributions
    opensource = np.random.choice([0, 1, 2, 3, 4, 5], n_students, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
    
    # Technical blogs/articles written
    blogs = np.random.choice([0, 1, 2, 3, 4], n_students, p=[0.7, 0.15, 0.1, 0.04, 0.01])
    
    # ==========================================
    # APTITUDE & SOFT SKILLS
    # ==========================================
    
    print("🎯 Generating aptitude and soft skills...")
    
    # Quantitative aptitude (0-100)
    quant_aptitude = overall_cgpa * 9 + np.random.normal(10, 12, n_students)
    quant_aptitude = np.clip(quant_aptitude, 30, 100).round(1)
    
    # Logical reasoning (0-100)
    logical = overall_cgpa * 8.5 + np.random.normal(15, 12, n_students)
    logical = np.clip(logical, 30, 100).round(1)
    
    # Verbal ability (0-100)
    verbal = np.random.normal(65, 15, n_students)
    verbal = np.clip(verbal, 30, 100).round(1)
    
    # Technical knowledge (0-100)
    technical = overall_cgpa * 9 + np.random.normal(5, 10, n_students)
    technical = np.clip(technical, 35, 100).round(1)
    
    # Coding test score (0-100)
    coding = overall_cgpa * 8 + internships * 5 + np.random.normal(10, 12, n_students)
    coding = np.clip(coding, 25, 100).round(1)
    
    # Communication skills (1-10)
    communication = np.random.normal(6.5, 1.8, n_students)
    communication = np.clip(communication, 3, 10).round(1)
    
    # Leadership score (1-10)
    leadership = participation * 0.8 + np.random.normal(1, 1.5, n_students)
    leadership = np.clip(leadership, 2, 10).round(1)
    
    # Teamwork score (1-10)
    teamwork = np.random.normal(7, 1.5, n_students)
    teamwork = np.clip(teamwork, 3, 10).round(1)
    
    # ==========================================
    # PLACEMENT READINESS
    # ==========================================
    
    print("💼 Generating placement readiness data...")
    
    # Resume score (1-10)
    resume_score = (overall_cgpa + internships * 2 + certifications * 0.5 + papers) / 2
    resume_score = np.clip(resume_score, 3, 10).round(1)
    
    # Mock interview performance (0-100)
    interview_score = (quant_aptitude + logical + verbal + communication * 10) / 4
    interview_score = interview_score.round(1)
    
    # Aptitude test attempts
    aptitude_attempts = np.random.choice([0, 1, 2, 3, 4, 5], n_students, p=[0.15, 0.2, 0.3, 0.2, 0.1, 0.05])
    
    # Companies applied to
    companies_applied = np.where(overall_cgpa >= 6.5,
                                np.random.randint(5, 25, n_students),
                                np.random.randint(0, 10, n_students))
    
    # ==========================================
    # SOCIOECONOMIC FACTORS
    # ==========================================
    
    print("🏠 Generating socioeconomic data...")
    
    family_income = np.random.choice(['<2L', '2-5L', '5-10L', '10-20L', '>20L'],
                                    n_students, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    
    parent_education = np.random.choice(
        ['10th or below', '12th', 'Graduate', 'Post-Graduate', 'Professional'],
        n_students, p=[0.20, 0.25, 0.30, 0.15, 0.10])
    
    siblings_in_college = np.random.choice([0, 1, 2], n_students, p=[0.6, 0.3, 0.1])
    
    distance_from_college = np.random.choice(['<5km', '5-15km', '15-30km', '>30km'],
                                             n_students, p=[0.25, 0.35, 0.25, 0.15])
    
    accommodation = np.random.choice(['Hostel', 'Day Scholar', 'PG'], 
                                    n_students, p=[0.35, 0.50, 0.15])
    
    scholarship = np.random.choice(['Yes', 'No'], n_students, p=[0.25, 0.75])
    
    # ==========================================
    # TARGET VARIABLES & PREDICTIONS
    # ==========================================
    
    print("🎯 Calculating target variables...")
    
    # 1. Graduation Status
    graduation_status = np.where(
        (overall_cgpa >= 6.5) & (current_backlogs == 0),
        'Clear',
        np.where(
            (overall_cgpa >= 5.5) & (current_backlogs <= 3),
            'At Risk',
            'Critical'
        )
    )
    
    # 2. Placement Status & Package
    placement_score = (
        overall_cgpa * 10 +
        internships * 15 +
        quant_aptitude * 0.3 +
        communication * 5 +
        certifications * 3
    )
    
    placement_prob = 1 / (1 + np.exp(-(placement_score - 100) / 20))  # Sigmoid
    is_placed = np.random.random(n_students) < placement_prob
    
    placement_status = np.where(is_placed, 'Placed', 'Not Placed')
    
    # Package (for placed students)
    package = np.where(
        is_placed,
        overall_cgpa * 0.9 + quant_aptitude * 0.04 + internships * 0.3 + np.random.normal(2, 1, n_students),
        np.nan
    )
    package = np.clip(package, 3.5, 15.0).round(1)
    
    # 3. Placement Prediction Category
    placement_prediction = np.where(
        (overall_cgpa >= 7.5) & (internships >= 1) & (current_backlogs == 0),
        'High',
        np.where(
            (overall_cgpa >= 6.5) & (current_backlogs <= 2),
            'Medium',
            'Low'
        )
    )
    
    # 4. Risk Score (0-100, higher = more risk)
    risk_score = (
        (10 - overall_cgpa) * 8 +
        current_backlogs * 6 +
        (100 - overall_attendance) * 0.25 +
        (100 - assignment_rate) * 0.2 +
        (10 - participation) * 2 +
        (3 - internships) * 3
    )
    risk_score = np.clip(risk_score, 0, 100).round(1)
    
    # 5. Dropout Risk
    dropout_risk = np.where(
        (overall_cgpa < 5.5) & (current_backlogs > 5),
        'High',
        np.where(
            (overall_cgpa < 6.5) & (current_backlogs > 3),
            'Medium',
            'Low'
        )
    )
    
    # ==========================================
    # CREATE DATAFRAME
    # ==========================================
    
    print("\n📋 Creating final dataset...")
    
    df = pd.DataFrame({
        # Basic Info
        'student_id': student_ids,
        'name': names,
        'gender': genders,
        
        # Semester-wise CGPA (Time-series)
        **sem_cgpas,
        
        # Semester-wise Attendance
        **sem_attendance,
        
        # Semester-wise Backlogs
        **sem_backlogs,
        
        # Overall Academic
        'overall_cgpa': overall_cgpa,
        'overall_attendance': overall_attendance,
        'total_backlogs_history': total_backlogs,
        'current_backlogs': current_backlogs,
        
        # Engagement
        'assignment_submission_rate': assignment_rate,
        'ontime_submission_rate': ontime_rate,
        'late_submissions_count': late_submissions,
        'quiz_average': quiz_avg,
        'lab_performance': lab_performance,
        'lab_attendance': lab_attendance,
        'project_score': project_score,
        'class_participation': participation,
        
        # Digital Engagement
        'lms_logins_per_week': lms_logins,
        'lms_time_hours_per_week': lms_time,
        'video_completion_rate': video_completion,
        'forum_posts': forum_posts,
        'resource_downloads': resource_downloads,
        
        # Study Patterns
        'study_hours_per_week': study_hours,
        'library_visits_per_week': library_visits,
        'study_group_frequency': study_group,
        'peak_study_time': study_time,
        
        # Activities
        'internships_completed': internships,
        'internship_rating': internship_rating,
        'certifications': certifications,
        'papers_presented': papers,
        'hackathons_participated': hackathons,
        'competitions_won': competitions_won,
        'opensource_contributions': opensource,
        'technical_blogs': blogs,
        
        # Aptitude & Skills
        'quantitative_aptitude': quant_aptitude,
        'logical_reasoning': logical,
        'verbal_ability': verbal,
        'technical_knowledge': technical,
        'coding_test_score': coding,
        'communication_skills': communication,
        'leadership_score': leadership,
        'teamwork_score': teamwork,
        
        # Placement Readiness
        'resume_score': resume_score,
        'mock_interview_score': interview_score,
        'aptitude_test_attempts': aptitude_attempts,
        'companies_applied': companies_applied,
        
        # Socioeconomic
        'family_income': family_income,
        'parent_education': parent_education,
        'siblings_in_college': siblings_in_college,
        'distance_from_college': distance_from_college,
        'accommodation': accommodation,
        'scholarship': scholarship,
        
        # Targets
        'graduation_status': graduation_status,
        'placement_status': placement_status,
        'package_lpa': package,
        'placement_prediction': placement_prediction,
        'risk_score': risk_score,
        'dropout_risk': dropout_risk
    })
    
    return df


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # Generate dataset
    df = generate_advanced_btech_dataset(n_students=300)
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/btech_ece_advanced.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Students: {len(df)}")
    print(f"   Total Features: {len(df.columns)}")
    print(f"   File Size: {os.path.getsize('data/btech_ece_advanced.csv') / 1024:.1f} KB")
    
    print(f"\n🎓 Graduation Status:")
    print(df['graduation_status'].value_counts())
    
    print(f"\n💼 Placement Status:")
    print(df['placement_status'].value_counts())
    print(f"   Placement Rate: {(df['placement_status'] == 'Placed').mean() * 100:.1f}%")
    print(f"   Average Package: {df['package_lpa'].mean():.2f} LPA")
    
    print(f"\n⚠️ Risk Distribution:")
    print(f"   High Risk (>60): {(df['risk_score'] > 60).sum()} students")
    print(f"   Medium Risk (30-60): {((df['risk_score'] >= 30) & (df['risk_score'] <= 60)).sum()} students")
    print(f"   Low Risk (<30): {(df['risk_score'] < 30).sum()} students")
    
    print(f"\n📋 Sample Data:")
    print(df[['student_id', 'overall_cgpa', 'risk_score', 'graduation_status', 'placement_status']].head(5))
    
    print("\n" + "="*70)
    print("✅ SAVED TO: data/btech_ece_advanced.csv")
    print("="*70)
    print("\n🚀 NEXT STEP: Run Phase 2 - Deep Learning Model Training")
    print("   Command: python phase2_deep_learning.py")
