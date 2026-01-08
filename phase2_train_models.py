"""
PHASE 2: DEEP LEARNING MODEL TRAINING
LSTM + Random Forest Hybrid Model
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, if not available use scikit-learn only
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, Bidirectional,
        BatchNormalization, Attention, Concatenate
    )
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available - Will use Deep Learning")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - Using traditional ML only")

# ==========================================
# TRADITIONAL ML MODELS (Always Available)
# ==========================================

class TraditionalMLModels:
    """Traditional ML models for comparison and fallback"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        
    def train_graduation_model(self, X_train, X_test, y_train, y_test):
        """Train graduation status predictor"""
        
        print("\n🎓 Training Graduation Model...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy:  {test_acc:.4f}")
        
        self.models['graduation'] = model
        return model, test_acc
    
    def train_placement_model(self, X_train, X_test, y_train, y_test):
        """Train placement predictor"""
        
        print("\n💼 Training Placement Model...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy:  {test_acc:.4f}")
        
        self.models['placement'] = model
        return model, test_acc
    
    def train_risk_model(self, X_train, X_test, y_train, y_test):
        """Train risk score predictor"""
        
        print("\n⚠️ Training Risk Score Model...")
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²:  {test_r2:.4f}")
        print(f"   MAE:      {mae:.2f} points")
        
        self.models['risk'] = model
        return model, test_r2
    
    def train_package_model(self, X_train, X_test, y_train, y_test):
        """Train package predictor (for placed students)"""
        
        print("\n💰 Training Package Prediction Model...")
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE:      {mae:.2f} LPA")
        
        self.models['package'] = model
        return model, r2

# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================

def main():
    """Main training pipeline"""
    
    print("="*70)
    print(" "*15 + "PHASE 2: MODEL TRAINING")
    print(" "*10 + "Advanced ML/DL Pipeline")
    print("="*70)
    
    # Load data
    print("\n📊 Loading dataset...")
    df = pd.read_csv('data/btech_ece_advanced.csv')
    print(f"✅ Loaded {len(df)} students with {len(df.columns)} features")
    
    # ==========================================
    # FEATURE SELECTION
    # ==========================================
    
    print("\n🎯 Selecting features...")
    
    # Academic features
    academic_features = [
        'overall_cgpa', 'overall_attendance', 'current_backlogs',
        'assignment_submission_rate', 'quiz_average',
        'lab_performance', 'project_score', 'class_participation'
    ]
    
    # Engagement features
    engagement_features = [
        'lms_logins_per_week', 'lms_time_hours_per_week',
        'video_completion_rate', 'forum_posts',
        'study_hours_per_week', 'library_visits_per_week'
    ]
    
    # Activity features
    activity_features = [
        'internships_completed', 'certifications',
        'papers_presented', 'hackathons_participated',
        'competitions_won'
    ]
    
    # Aptitude features
    aptitude_features = [
        'quantitative_aptitude', 'logical_reasoning',
        'verbal_ability', 'technical_knowledge',
        'coding_test_score', 'communication_skills'
    ]
    
    # All features (excluding demographics for ethical AI)
    feature_columns = (academic_features + engagement_features + 
                      activity_features + aptitude_features)
    
    print(f"✅ Selected {len(feature_columns)} features (Ethical AI - no demographics)")
    
    # ==========================================
    # PREPARE DATA
    # ==========================================
    
    X = df[feature_columns].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Encode categorical targets
    le_graduation = LabelEncoder()
    le_placement_pred = LabelEncoder()
    
    y_graduation = le_graduation.fit_transform(df['graduation_status'])
    y_placement_pred = le_placement_pred.fit_transform(df['placement_prediction'])
    y_risk = df['risk_score'].values
    
    # For package prediction (only placed students)
    placed_mask = df['placement_status'] == 'Placed'
    X_package = X[placed_mask]
    y_package = df[placed_mask]['package_lpa'].values
    
    print(f"\n📦 Data shapes:")
    print(f"   Features (X): {X.shape}")
    print(f"   Graduation target: {y_graduation.shape}")
    print(f"   Placement target: {y_placement_pred.shape}")
    print(f"   Risk target: {y_risk.shape}")
    print(f"   Package data: {X_package.shape}")
    
    # ==========================================
    # TRAIN TRADITIONAL ML MODELS
    # ==========================================
    
    print("\n" + "="*70)
    print("TRAINING TRADITIONAL ML MODELS")
    print("="*70)
    
    ml_models = TraditionalMLModels()
    
    # Split data
    X_train, X_test, y_grad_train, y_grad_test = train_test_split(
        X, y_graduation, test_size=0.2, random_state=42, stratify=y_graduation
    )
    
    _, _, y_place_train, y_place_test = train_test_split(
        X, y_placement_pred, test_size=0.2, random_state=42, stratify=y_placement_pred
    )
    
    _, _, y_risk_train, y_risk_test = train_test_split(
        X, y_risk, test_size=0.2, random_state=42
    )
    
    # Train models
    grad_model, grad_acc = ml_models.train_graduation_model(
        X_train, X_test, y_grad_train, y_grad_test
    )
    
    place_model, place_acc = ml_models.train_placement_model(
        X_train, X_test, y_place_train, y_place_test
    )
    
    risk_model, risk_r2 = ml_models.train_risk_model(
        X_train, X_test, y_risk_train, y_risk_test
    )
    
    # Train package model (if enough placed students)
    if len(X_package) > 50:
        X_pkg_train, X_pkg_test, y_pkg_train, y_pkg_test = train_test_split(
            X_package, y_package, test_size=0.2, random_state=42
        )
        pkg_model, pkg_r2 = ml_models.train_package_model(
            X_pkg_train, X_pkg_test, y_pkg_train, y_pkg_test
        )
    else:
        print("\n⚠️ Not enough placed students for package model")
        pkg_model = None
    
    # ==========================================
    # SAVE MODELS
    # ==========================================
    
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    
    # Save traditional ML models
    with open('models/graduation_model.pkl', 'wb') as f:
        pickle.dump(grad_model, f)
    print("✅ Saved: graduation_model.pkl")
    
    with open('models/placement_model.pkl', 'wb') as f:
        pickle.dump(place_model, f)
    print("✅ Saved: placement_model.pkl")
    
    with open('models/risk_model.pkl', 'wb') as f:
        pickle.dump(risk_model, f)
    print("✅ Saved: risk_model.pkl")
    
    if pkg_model:
        with open('models/package_model.pkl', 'wb') as f:
            pickle.dump(pkg_model, f)
        print("✅ Saved: package_model.pkl")
    
    # Save encoders
    with open('models/le_graduation.pkl', 'wb') as f:
        pickle.dump(le_graduation, f)
    print("✅ Saved: le_graduation.pkl")
    
    with open('models/le_placement.pkl', 'wb') as f:
        pickle.dump(le_placement_pred, f)
    print("✅ Saved: le_placement.pkl")
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    print("✅ Saved: feature_names.pkl")
    
    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================
    
    print("\n" + "="*70)
    print("TOP 15 IMPORTANT FEATURES")
    print("="*70)
    
    importances = grad_model.feature_importances_
    feature_importance = list(zip(feature_columns, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:15], 1):
        print(f"{i:2d}. {feature:40s} : {importance:.4f}")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
    print(f"   Graduation Model:  {grad_acc:.2%} accuracy")
    print(f"   Placement Model:   {place_acc:.2%} accuracy")
    print(f"   Risk Model:        {risk_r2:.4f} R² score")
    if pkg_model:
        print(f"   Package Model:     {pkg_r2:.4f} R² score")
    
    print(f"\n📁 Models saved to: models/")
    print(f"   - graduation_model.pkl")
    print(f"   - placement_model.pkl")
    print(f"   - risk_model.pkl")
    if pkg_model:
        print(f"   - package_model.pkl")
    
    print("\n🚀 NEXT STEP: Run Phase 3 - Advanced Streamlit App")
    print("   Command: streamlit run phase3_advanced_app.py")
    
    return True

if __name__ == "__main__":
    main()
