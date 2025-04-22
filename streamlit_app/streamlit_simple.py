# Save this as 'streamlit_simple.py' in your project root directory
import streamlit as st
import pandas as pd
import numpy as np
import os

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="PopeCast 2025",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data
def load_cardinal_data(path="data/cardinals_scored.csv"):
    """Load scored cardinal data."""
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return None
        
    df = pd.read_csv(path)
    df = df[df["age"] < 100]  # Sanity filter
    return df

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Coat_of_arms_Holy_See.svg/240px-Coat_of_arms_Holy_See.svg.png", width=100)
st.sidebar.title("PopeCast 2025")
st.sidebar.caption("Papal Conclave Prediction System")

# Filter settings
st.sidebar.header("🧮 Cardinal Filters")
df = load_cardinal_data()

if df is None:
    st.error("Could not load cardinal data. Please make sure data/cardinals_scored.csv exists.")
    st.stop()

electors = df[df["age"] < 80].copy()  # Only eligible electors

min_age, max_age = int(electors["age"].min()), int(electors["age"].max())
age_range = st.sidebar.slider(
    "Age Range", 
    min_value=min_age, 
    max_value=max_age, 
    value=(min_age, max_age)
)

countries = sorted(electors["country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Countries", 
    countries, 
    default=countries
)

# Sort options
sort_option = st.sidebar.selectbox(
    "Sort by", 
    ["PPI Score", "PVS Score", "Age", "Country"]
)

# Apply filters
filtered_electors = electors.copy()
filtered_electors = filtered_electors[
    (filtered_electors["age"] >= age_range[0]) & 
    (filtered_electors["age"] <= age_range[1]) &
    (filtered_electors["country"].isin(selected_countries))
]

# -------------------------------
# Main Dashboard
# -------------------------------
st.title("📊 PopeCast 2025: Papal Conclave Leaderboard")

# Top section with key stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📝 Total Cardinals", len(df))
with col2:
    st.metric("🗳️ Eligible Electors", len(electors))
with col3:
    st.metric("⚖️ Required Majority", int(np.ceil(2/3 * len(filtered_electors))))

# Leaderboard section
st.header("Cardinal Leaderboard")
st.markdown("Explore cardinals by their pope potential (PPI) or papal vibes (PVS).")

# Sort data based on selection
if sort_option == "PPI Score":
    score_col = "enhanced_ppi" if "enhanced_ppi" in filtered_electors.columns else "ppi_score"
    sorted_df = filtered_electors.sort_values(score_col, ascending=False)
elif sort_option == "PVS Score":
    sorted_df = filtered_electors.sort_values("pvs_score", ascending=False)
elif sort_option == "Age":
    sorted_df = filtered_electors.sort_values("age")
else:  # Country
    sorted_df = filtered_electors.sort_values("country")

# Prepare columns for display
display_columns = ["name", "age", "country", "current_role"]

# Add available scores
if "enhanced_ppi" in filtered_electors.columns:
    display_columns.append("enhanced_ppi")
    score_column_name = "Enhanced PPI"
elif "ppi_score" in filtered_electors.columns:
    display_columns.append("ppi_score")
    score_column_name = "PPI"
    
if "pvs_score" in filtered_electors.columns:
    display_columns.append("pvs_score")

# Map column names for display
column_map = {
    "name": "Cardinal",
    "age": "Age",
    "country": "Country",
    "current_role": "Role",
    "enhanced_ppi": score_column_name,
    "ppi_score": "PPI",
    "pvs_score": "PVS",
}

# Select and rename columns that exist
existing_columns = [col for col in display_columns if col in sorted_df.columns]
display_df = sorted_df[existing_columns].rename(columns={col: column_map.get(col, col) for col in existing_columns})

# Display the table
st.dataframe(display_df, use_container_width=True)
st.caption("Click column headers to sort. Cardinals must be under 80 to vote in conclave.")

# Footer
st.markdown("---")
st.caption("© PopeCast 2025 • Created for research and educational purposes")
