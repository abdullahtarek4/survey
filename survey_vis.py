import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Academic Readiness Dashboard", layout="wide")
st.title("Academic Readiness Dashboard")

# Sidebar page selection
page = st.sidebar.selectbox("Select a visualization", [
    "📊 Bar Chart - Career Services vs Academic Level",
    "👩‍🦱👨‍🦱 Pie Chart - Gender vs Readiness",
    "📈 Histogram - Current CGPA Distribution",
    "🎓 Bar Chart - Certificates vs Readiness",
    "💻🏢 Bar Chart - Internship Types by Department",
    "🟢🟡Multi-variate Bubble Chart - Job Applications vs CGPA vs Readiness",
    "💼 Workshops vs Job Offers",
    "📉 Confidence vs Internships",
    "✨ Career Paths Preference",
    "📚 Top 3 Elective Courses",
    "🔝 Most Impactful External Courses"  
])

# File path
data_path = r"All_Merge_Shuffle_data.xlsx"

try:
    df = pd.read_excel(data_path)

    if "What is your department?" in df.columns:
        department_options = ["All Departments"] + sorted(df["What is your department?"].dropna().unique().tolist())
        selected_dept = st.selectbox("Select Department", department_options, key=page)
        filtered_df = df if selected_dept == "All Departments" else df[df["What is your department?"] == selected_dept]
        def plot_small(fig):
            fig.set_size_inches(8, 5)
            st.pyplot(fig)
        if page == "📊 Bar Chart - Career Services vs Academic Level":
            st.subheader("📊 Career Services Help vs Academic Level")
            col1 = "To what extent do you think your university’s career services have helped you prepare for the job market?"
            col2 = "What is your current academic level?"
            if col1 in filtered_df.columns and col2 in filtered_df.columns:
                group_data = filtered_df.groupby([col2, col1]).size().unstack().fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                custom_colors = {"Not at all": '#6fa3f4', "Slightly": '#76c793', "Moderately": '#f39c11', "Very much": '#9b59b6'}
                colors = [custom_colors.get(val, "#cccccc") for val in group_data.columns]
                group_data.plot(kind="bar", ax=ax, color=colors)
                ax.set_xlabel("Academic Level")
                ax.set_ylabel("Count")
                ax.set_title(f"Career Services Help vs Academic Level - {selected_dept}")
                plt.xticks(rotation=45)
                plot_small(fig)
            else:
                st.warning("⚠ Required columns not found.")

        elif page == "👩‍🦱👨‍🦱 Pie Chart - Gender vs Readiness":
            st.subheader("👩‍🦱👨‍🦱 Pie Chart - Gender vs Readiness")
            if "What is your gender?" in filtered_df.columns and "Readiness_Status" in filtered_df.columns:
                combo = filtered_df.groupby(["What is your gender?", "Readiness_Status"]).size().reset_index(name='count')
                labels = [f"{row['What is your gender?']} - {row['Readiness_Status']}" for _, row in combo.iterrows()]
                counts = combo['count'].values
                fig, ax = plt.subplots()
                gender_ready_colors = [
                    '#f582a1' if gender == 'Female' and readiness == 'Ready' else
                    '#f0a6b2' if gender == 'Female' and readiness == 'Not Ready' else
                    '#6a9fd9' if gender == 'Male' and readiness == 'Ready' else
                    '#a8c9f4'
                    for gender, readiness in zip(combo['What is your gender?'], combo['Readiness_Status'])
                ]
                ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops={"edgecolor": "black"}, colors=gender_ready_colors)
                ax.set_title(f"Gender vs Readiness - {selected_dept}")
                plot_small(fig)
            else:
                st.warning("⚠ Required columns not found.")

        elif page == "📈 Histogram - Current CGPA Distribution":
            st.subheader("📈 CGPA Distribution")
            cgpa_col = "What is your current CGPA?"
            if cgpa_col in filtered_df.columns:
                clean_df = filtered_df[pd.to_numeric(filtered_df[cgpa_col], errors='coerce').notna()]
                clean_df[cgpa_col] = clean_df[cgpa_col].astype(float)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(clean_df[cgpa_col], bins=20, kde=True, color="#A0D9B8", ax=ax)
                ax.set_xlabel("CGPA")
                ax.set_ylabel("Number of Students")
                ax.set_title(f"CGPA Distribution - {selected_dept}")
                plot_small(fig)
            else:
                st.warning("⚠ 'What is your current CGPA?' column not found.")

        elif page == "🎓 Bar Chart - Certificates vs Readiness":
            st.subheader("🎓 Certificates Achieved vs Readiness Status")
            if "How many certificates have you achieved so far?" in filtered_df.columns and "Readiness_Status" in filtered_df.columns:
                group_data = filtered_df.groupby(["How many certificates have you achieved so far?", "Readiness_Status"]).size().unstack().fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                group_data.plot(kind="bar", ax=ax, color=["#76c793", "#f39c11"])
                ax.set_xlabel("Number of Certificates")
                ax.set_ylabel("Count")
                ax.set_title(f"Certificates Achieved vs Readiness - {selected_dept}")
                plt.xticks(rotation=45)
                plot_small(fig)
            else:
                st.warning("⚠ Required columns not found.")

        elif page == "💻🏢 Bar Chart - Internship Types by Department":
            st.subheader("💻🏢 Bar Chart - Internship Types by Department")
            types = [
                "Which types of internships have you completed during your studies, and how many for each type?  [Virtual/Remote Internship]",
                "Which types of internships have you completed during your studies, and how many for each type?  [Industry/Corporate Internship]",
                "Which types of internships have you completed during your studies, and how many for each type?  [Government Internship]"
            ]
            shortened_labels = ["Virtual /Remote Internership", "Industry /Corporate Internership", "Government Internership"]
            if all(col in filtered_df.columns for col in types):
                internship_data = filtered_df[types].apply(pd.to_numeric, errors='coerce').sum(axis=0)
                fig, ax = plt.subplots(figsize=(10, 6))
                internship_data.plot(kind="bar", ax=ax, color=['#A8D0E6', '#B9E4C9', '#F7D98C'])
                ax.set_xlabel("Internship Type")
                ax.set_ylabel("Total Count")
                ax.set_title(f"Internship Types vs Count - {selected_dept}")
                ax.set_xticklabels(shortened_labels, rotation=45, fontsize=8)  
                plt.yticks(fontsize=8)
                plot_small(fig)
            else:
                st.warning("⚠ Required columns not found.")

        

        elif page == "🟢🟡Multi-variate Bubble Chart - Job Applications vs CGPA vs Readiness":
            st.subheader("🟢🟡Multi-variate Bubble Chart - Job Applications vs CGPA vs Readiness")
            if "What is your current CGPA?" in filtered_df.columns and "In the past 6 months, how many job applications have you submitted?" in filtered_df.columns and "Career_Readiness_Percentage" in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    filtered_df["What is your current CGPA?"], 
                    filtered_df["In the past 6 months, how many job applications have you submitted?"], 
                    s=filtered_df["Career_Readiness_Percentage"] * 10, 
                    c=filtered_df["Career_Readiness_Percentage"], 
                    cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5
                )
                ax.set_xlabel("CGPA")
                ax.set_ylabel("Job Applications Submitted")
                ax.set_title(f"Job Applications vs CGPA vs Readiness - {selected_dept}")
                fig.colorbar(scatter, label="Career Readiness Percentage")
                plot_small(fig)
            else:
                st.warning("⚠ Required columns not found.")

        

        elif page == "💼 Workshops vs Job Offers":
            st.subheader("💼 Workshops vs Job Offers")
            workshop_col = "Have you attended any career-related workshops or training sessions?"
            offer_col = "Have you received any job offers before graduation?"
            if workshop_col in filtered_df.columns and offer_col in filtered_df.columns:
                group_data = filtered_df.groupby([workshop_col, offer_col]).size().unstack().fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                group_data.plot(kind="bar", ax=ax, color=["#f0a6b2", "#4c72b0"])
                ax.set_xlabel("Number of Workshops")
                ax.set_ylabel("Count")
                ax.set_title(f"Workshops vs Job Offers - {selected_dept}")
                plt.xticks(rotation=45)
                plot_small(fig)
            else:
                st.warning("⚠ Required columns not found.")
                
        elif page == "📉 Confidence vs Internships":
            st.subheader("📉 Confidence in Securing Job vs Number of Internships")
            confidence_col = "On a scale of 0 to 5, how confident are you in your ability to secure a job after graduation, considering your skills, experience, and job market conditions?"
            internships_col = "How many internships have you completed during your studies?"

            # Convert columns to numeric and filter only students with >0 internships
            filtered_df[confidence_col] = pd.to_numeric(filtered_df[confidence_col], errors='coerce')
            filtered_df[internships_col] = pd.to_numeric(filtered_df[internships_col], errors='coerce')
            filtered_df = filtered_df[(filtered_df[internships_col] > 0)]

            if confidence_col in filtered_df.columns and internships_col in filtered_df.columns:
                # Group data and calculate counts
                group_data = filtered_df.groupby([confidence_col, internships_col]).size().reset_index(name="Count")

                # Create bar plot with soft colors
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=confidence_col, y="Count", data=group_data, ax=ax, hue=internships_col, palette="muted")

                ax.set_xlabel("Confidence in Securing Job (0-5)")
                ax.set_ylabel("Number of Students (Grouped by Internships)")
                ax.set_title(f"Confidence vs Internships (Internships > 0) - {selected_dept}")
                plt.xticks(rotation=0)
                plot_small(fig)
            else:
                st.warning("⚠ Required columns not found.")

                
        elif page == "✨ Career Paths Preference":
            st.subheader("✨ Career Paths Preference")
            col = "Which of the following career paths do you prefer?"
            if col in filtered_df.columns:
                all_paths = filtered_df[col].dropna().astype(str).str.split(',').explode().str.strip()
                top_paths = all_paths.value_counts().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_paths.values, y=top_paths.index, ax=ax, palette="pastel")
                ax.set_xlabel("Number of Students")
                ax.set_ylabel("Career Path")
                ax.set_title(f"Preferred Career Paths - {selected_dept}")
                plot_small(fig)
            else:
                st.warning("⚠ Column not found.")
                
        elif page == "📚 Top 3 Elective Courses":
            st.subheader("📚 Top 3 Elective Courses")
            elective_col = "Which elective course have you found most helpful for your career preparation and readiness?"
            if elective_col in filtered_df.columns:
                elective_data = filtered_df[elective_col].str.split(",", expand=True).stack().value_counts().head(3)
                fig, ax = plt.subplots(figsize=(10, 6))
                elective_data.plot(kind="bar", ax=ax, color=sns.color_palette("pastel", len(elective_data)))
                ax.set_xlabel("Elective Course")
                ax.set_ylabel("Count")
                ax.set_title(f"Top 3 Elective Courses - {selected_dept}")
                plt.xticks(rotation=45)
                plot_small(fig)
            else:
                st.warning("⚠ 'Top 3 Elective Courses' column not found.")
        elif page == "🔝 Most Impactful External Courses":
           st.subheader("🔝 Most Impactful External Courses (Top 5)")

           # Assuming the column name is correct
           col = "Have you taken any courses outside of the university that you found particularly impactful that should be added to the university curriculum?"
    
           if col in filtered_df.columns:
           # Splitting the courses and counting the frequency of each course
             courses = filtered_df[col].dropna().str.split(",").explode().str.strip()
             courses = courses[~courses.isin(['Yes', 'No','yes'])]  # Removing "Yes" and "No"
        
             # Getting the top 5 most frequent courses
             top_courses = courses.value_counts().head(5)
        
             # Plotting the bar chart for the top 5 courses
             fig, ax = plt.subplots(figsize=(10, 6))
             sns.barplot(x=top_courses.values, y=top_courses.index, ax=ax, palette="pastel")
             ax.set_xlabel("Number of Students Who Selected")
             ax.set_ylabel("Course")
             ax.set_title("Top 5 Most Impactful External Courses")
             plot_small(fig)
           else:
             st.warning("⚠ 'Have you taken any courses outside of the university that you found particularly impactful that should be added to the university curriculum?' column not found.")
except Exception as e:
    st.error(f"Error loading data: {e}")


import joblib
from catboost import CatBoostRegressor
import os

# Load models
model_path = os.path.join(os.getcwd(), "best_model_GB.pkl")
gb_model = joblib.load(model_path)

cat_model = CatBoostRegressor()
model_path2 = os.path.join(os.getcwd(), "best_model_catboost.pkl")
cat_model.load_model("best_model_catboost.pkl")



# Input fields
st.header("📋 Enter Student Details")
study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=2.0)
attendance = st.slider("Attendance Percentage", min_value=0, max_value=100, value=75)
assignments = st.number_input("Assignments Completed", min_value=0, max_value=10, value=5)

# Collect inputs
input_data = pd.DataFrame([[study_hours, attendance, assignments]],
                          columns=["study_hours", "attendance", "assignments"])

# Predict button
if st.button("Predict Readiness"):
    gb_pred = gb_model.predict(input_data)[0]
    cat_pred = cat_model.predict(input_data)[0]

    # Threshold for readiness
    threshold = 0.5
    gb_result = "Ready ✅" if gb_pred >= threshold else "Not Ready ❌"
    cat_result = "Ready ✅" if cat_pred >= threshold else "Not Ready ❌"

    st.subheader("🔮 Prediction Results:")
    st.write(f"**Gradient Boosting Prediction:** {gb_result} (Score: {gb_pred:.2f})")
    st.write(f"**CatBoost Prediction:** {cat_result} (Score: {cat_pred:.2f})")
