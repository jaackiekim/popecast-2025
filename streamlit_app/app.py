import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.simulate_conclave import (
    load_eligible_cardinals, 
    run_realistic_conclave,
    run_monte_carlo_simulation,
    plot_election_probabilities,
    plot_conclave_progression
)

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
    df = pd.read_csv(path)
    df = df[df["age"] < 100]  # Sanity filter
    return df

@st.cache_data
def run_simulation(_electors, num_sims=500):
    """Run Monte Carlo simulation with caching."""
    sim_stats = run_monte_carlo_simulation(
        _electors.copy(), 
        num_simulations=num_sims,
        verbose=False
    )
    return sim_stats

def run_single_conclave(_electors):
    """Run a single conclave simulation."""
    winner, voting_history, stats = run_realistic_conclave(
        _electors.copy(),
        verbose=False
    )
    return winner, voting_history, stats

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Coat_of_arms_Holy_See.svg/240px-Coat_of_arms_Holy_See.svg.png", width=100)
st.sidebar.title("PopeCast 2025")
st.sidebar.caption("Papal Conclave Prediction System")

# Data loading options
data_path = st.sidebar.selectbox(
    "Cardinal Data Source",
    ["data/cardinals_scored.csv", "data/cardinals_enriched.csv"],
    index=0
)

# Simulation settings
st.sidebar.header("🔮 Simulation Settings")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 500)

# Context settings
st.sidebar.header("🌍 Conclave Context")
regional_trend = st.sidebar.selectbox(
    "Regional Emphasis",
    ["global", "growth", "tradition"],
    index=0,
    help="'global' is balanced, 'growth' favors Global South, 'tradition' favors Europe"
)

theological_trend = st.sidebar.slider(
    "Theological Center", 
    1.0, 10.0, 6.2, 0.1,
    help="1=Very Conservative, 10=Very Progressive"
)

# Filter settings
st.sidebar.header("🧮 Cardinal Filters")
df = load_cardinal_data(data_path)
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

if "theological_position" in electors.columns:
    theo_range = st.sidebar.slider(
        "Theological Position", 
        1.0, 10.0, (1.0, 10.0), 0.5,
        help="1=Very Conservative, 10=Very Progressive"
    )
else:
    theo_range = (1.0, 10.0)

# Apply filters
filtered_electors = electors.copy()
filtered_electors = filtered_electors[
    (filtered_electors["age"] >= age_range[0]) & 
    (filtered_electors["age"] <= age_range[1]) &
    (filtered_electors["country"].isin(selected_countries))
]

if "theological_position" in filtered_electors.columns:
    filtered_electors = filtered_electors[
        (filtered_electors["theological_position"] >= theo_range[0]) &
        (filtered_electors["theological_position"] <= theo_range[1])
    ]

# Normalize weights for filtered electors
if len(filtered_electors) > 0:
    filtered_electors["weight"] = filtered_electors["enhanced_ppi"] if "enhanced_ppi" in filtered_electors.columns else filtered_electors["ppi_score"]
    filtered_electors["weight"] = filtered_electors["weight"] / filtered_electors["weight"].sum()

# Run button
run_sim = st.sidebar.button("🔄 Run Prediction", type="primary")

# -------------------------------
# Main Dashboard
# -------------------------------
st.title("📊 PopeCast 2025: Papal Conclave Prediction")

# Top section with key stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📝 Total Cardinals", len(df))
with col2:
    st.metric("🗳️ Eligible Electors", len(electors))
with col3:
    st.metric("⚖️ Required Majority", int(np.ceil(2/3 * len(filtered_electors))))

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Leaderboard", "Simulation", "About"])

with tab1:
    st.header("Papal Election Prediction")
    
    if len(filtered_electors) == 0:
        st.warning("⚠️ No cardinals match your filter criteria. Please adjust your filters.")
    elif len(filtered_electors) < 10:
        st.warning("⚠️ Very few cardinals match your criteria. Results may be unreliable.")
    elif run_sim:
        with st.spinner("Running papal conclave simulations..."):
            # Run Monte Carlo simulation
            sim_stats = run_simulation(filtered_electors, num_simulations)
            
            # Display top prediction
            st.subheader("Most Likely Pope")
            most_likely = sim_stats["most_likely_pope"]
            probability = sim_stats["election_probability"] * 100
            
            if most_likely:
                # Get data about most likely pope
                pope_info = filtered_electors[filtered_electors["name"] == most_likely].iloc[0]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.info(f"### {most_likely}")
                    st.caption(f"**Country:** {pope_info['country']}")
                    st.caption(f"**Age:** {int(pope_info['age'])}")
                    st.caption(f"**Election Probability:** {probability:.1f}%")
                    
                with col2:
                    # Election probability chart
                    fig = plot_election_probabilities(sim_stats)
                    st.pyplot(fig)
                
                # Prediction details
                st.subheader("Prediction Details")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Conclave Days", f"{sim_stats['median_days']:.1f}")
                with col2:
                    st.metric("Expected Ballots", f"{sim_stats['median_ballots']:.1f}")
                with col3:
                    no_election = sim_stats['no_election_probability'] * 100
                    st.metric("No Election Probability", f"{no_election:.1f}%")
            else:
                st.error("No clear prediction could be made. Consider adjusting filters or increasing simulations.")
    else:
        st.info("👈 Adjust the simulation settings and click 'Run Prediction' to generate a forecast.")
        
        # Show sample top cardinals by PPI
        st.subheader("Top Cardinals by Pope Potential Index (PPI)")
        top_cardinals = filtered_electors.sort_values(
            "enhanced_ppi" if "enhanced_ppi" in filtered_electors.columns else "ppi_score", 
            ascending=False
        ).head(10)
        
        st.dataframe(
            top_cardinals[[
                "name", "age", "country", "current_role", 
                "enhanced_ppi" if "enhanced_ppi" in top_cardinals.columns else "ppi_score",
                "pvs_score"
            ]].rename(columns={
                "name": "Cardinal",
                "age": "Age",
                "country": "Country",
                "current_role": "Role",
                "enhanced_ppi": "PPI",
                "ppi_score": "PPI",
                "pvs_score": "PVS"
            }),
            use_container_width=True
        )

with tab2:
    st.header("Cardinal Leaderboard")
    st.markdown("Explore cardinals by their pope potential (PPI) or papal vibes (PVS).")
    
    # Sorting options
    sort_option = st.selectbox(
        "Sort by", 
        ["Enhanced PPI" if "enhanced_ppi" in filtered_electors.columns else "PPI Score", "PVS Score", "Age"]
    )
    
    # Sort data
    if sort_option in ["Enhanced PPI", "PPI Score"]:
        sorted_df = filtered_electors.sort_values(
            "enhanced_ppi" if "enhanced_ppi" in filtered_electors.columns else "ppi_score", 
            ascending=False
        )
    elif sort_option == "PVS Score":
        sorted_df = filtered_electors.sort_values("pvs_score", ascending=False)
    else:
        sorted_df = filtered_electors.sort_values("age")
    
    # Prepare columns for display
    display_columns = ["name", "age", "country", "current_role"]
    
    # Add available scores
    if "enhanced_ppi" in filtered_electors.columns:
        display_columns.append("enhanced_ppi")
    elif "ppi_score" in filtered_electors.columns:
        display_columns.append("ppi_score")
        
    if "pvs_score" in filtered_electors.columns:
        display_columns.append("pvs_score")
        
    # Add theological position if available
    if "theological_position" in filtered_electors.columns:
        display_columns.append("theological_position")
    
    # Add bio URL if available
    if "bio_url" in filtered_electors.columns:
        display_columns.append("bio_url")
    
    # Map column names for display
    column_map = {
        "name": "Cardinal",
        "age": "Age",
        "country": "Country",
        "current_role": "Role",
        "enhanced_ppi": "Enhanced PPI",
        "ppi_score": "PPI",
        "pvs_score": "PVS",
        "theological_position": "Theological Position",
        "bio_url": "Wikipedia"
    }
    
    display_df = sorted_df[display_columns].rename(columns=column_map)
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    st.caption("Click column headers to sort. Cardinals must be under 80 to vote in conclave.")

with tab3:
    st.header("Conclave Simulation")
    st.markdown("Run a single conclave simulation to see the voting dynamics.")
    
    run_single = st.button("🏁 Run Single Conclave")
    
    if run_single:
        with st.spinner("Simulating conclave..."):
            winner, voting_history, stats = run_single_conclave(filtered_electors)
            
            if winner:
                st.success(f"### {winner} Elected Pope")
                st.write(f"Election completed on Day {stats['winning_day']}, Ballot #{stats['winning_ballot']}")
                
                # Show vote progression
                st.subheader("Vote Progression")
                fig = plot_conclave_progression(voting_history, filtered_electors, winner)
                st.pyplot(fig)
                
                # Show dynamics
                st.subheader("Conclave Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Ballots", stats["total_ballots"])
                with col2:
                    final_votes = voting_history[-1].loc[winner]
                    vote_share = final_votes / len(filtered_electors) * 100
                    st.metric("Winning Vote Share", f"{vote_share:.1f}%")
                with col3:
                    st.metric("Days of Conclave", stats["days"])
                
                # Show top candidates by day
                st.subheader("Top Candidates by Day")
                for day, day_ballots in enumerate(stats["top_candidates_by_day"], 1):
                    st.write(f"**Day {day}:**")
                    ballot_text = ""
                    for i, ballot in enumerate(day_ballots, 1):
                        top_names = ', '.join(ballot[:min(3, len(ballot))])
                        ballot_text += f"Ballot {i}: {top_names} | "
                    st.text(ballot_text.rstrip(" |"))
            else:
                st.error("No pope elected after maximum conclave duration.")
    else:
        st.info("Click 'Run Single Conclave' to simulate a papal election.")

with tab4:
    st.header("About PopeCast 2025")
    st.markdown("""
    ### Papal Conclave Prediction System
    
    PopeCast 2025 is a computational model that predicts papal elections by simulating the conclave process.
    The system combines historical patterns, cardinal data analysis, and realistic voting dynamics to forecast
    the most likely outcomes of a papal conclave.
    
    #### Key Features:
    
    - **Data-Driven Analysis**: Comprehensive profile of each cardinal with predictive factors
    - **Realistic Conclave Simulation**: Models actual conclave voting procedures and dynamics
    - **Probabilistic Prediction**: Uses Monte Carlo simulation for robust forecasting
    - **Contextual Scenarios**: Allows exploration of different conclave contexts and trends
    
    #### Model Limitations:
    
    - Cannot account for divine inspiration or the movement of the Holy Spirit
    - Limited information about cardinals' private theological positions and relationships
    - Conclave deliberations are secret, so some dynamics remain speculative
    - The model is for educational and research purposes only
    
    #### Data Sources:
    
    - Cardinal biographical data from public sources
    - Historical conclave patterns from Church records
    - Voting dynamics based on accounts of past conclaves
    
    #### Disclaimer:
    
    This tool is created for educational and research purposes only. It represents a computational 
    approach to understanding the complex spiritual, theological, and institutional process of 
    papal elections.
    """)

# Footer
st.markdown("---")
st.caption("© PopeCast 2025 • Created for research and educational purposes")
