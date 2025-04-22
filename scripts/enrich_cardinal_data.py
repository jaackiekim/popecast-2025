import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime

# Load scraped data
def load_raw_data(csv_path="data/cardinals_raw.csv"):
    """Load and validate raw cardinal data."""
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} cardinals from raw data")
    return df

# === ENRICHMENT FUNCTIONS ===

def estimate_theological_position(df):
    """
    Estimate theological position on a scale of 1-10
    (1 = most conservative, 10 = most progressive)
    Uses role, statements, and appointing pope as proxies
    """
    # Base values by appointing pope (rough approximations)
    pope_ideology = {
        "John Paul II": 3.5,  # More conservative
        "Benedict XVI": 3.0,  # Most conservative
        "Francis": 6.5        # More progressive
    }
    
    # Default to neutral position
    df["theological_position"] = 5.0
    
    # Identify appointing pope based on cardinal creation date
    def get_appointing_pope(birthdate_raw):
        try:
            birthdate = datetime.strptime(birthdate_raw, "%d %B %Y")
            year = birthdate.year
            
            # Roughly estimate when they became cardinal based on age
            # (typically at least 50 years old)
            estimated_cardinal_year = max(year + 50, 1970)
            
            if estimated_cardinal_year <= 2005:
                return "John Paul II"
            elif estimated_cardinal_year <= 2013:
                return "Benedict XVI"
            else:
                return "Francis"
        except:
            return "Unknown"
    
    df["appointing_pope"] = df["birthdate_raw"].apply(get_appointing_pope)
    
    # Adjust base theological position by appointing pope
    for pope, ideology_score in pope_ideology.items():
        df.loc[df["appointing_pope"] == pope, "theological_position"] = ideology_score
    
    # Further refine based on role keywords
    conservative_keywords = ["tradition", "doctrine", "faith", "morals"]
    progressive_keywords = ["dialogue", "reform", "pastoral", "synod"]
    
    # Adjust scores based on roles (simplified for illustration)
    for i, cardinal in df.iterrows():
        role = str(cardinal["current_role"]).lower()
        
        # Shift toward conservative if role contains conservative keywords
        cons_matches = sum(1 for word in conservative_keywords if word in role)
        if cons_matches > 0:
            df.at[i, "theological_position"] -= cons_matches * 0.5
            
        # Shift toward progressive if role contains progressive keywords
        prog_matches = sum(1 for word in progressive_keywords if word in role)
        if prog_matches > 0:
            df.at[i, "theological_position"] += prog_matches * 0.5
    
    # Ensure values stay within 1-10 range
    df["theological_position"] = df["theological_position"].clip(1, 10)
    
    print(f"✅ Added theological position estimates")
    return df

def estimate_language_abilities(df):
    """
    Estimate language abilities based on country, education, and roles
    Returns a list of likely languages spoken
    """
    # Base language is native language from country
    language_map = {
        "Italy": ["Italian", "Latin"],
        "United States": ["English"],
        "Germany": ["German"],
        "France": ["French"],
        "Spain": ["Spanish"],
        "Brazil": ["Portuguese", "Spanish"],
        "India": ["English", "Hindi"],
        "Philippines": ["English", "Filipino"],
        "Nigeria": ["English"],
        "Poland": ["Polish"],
        # Add more countries as needed
    }
    
    # Default language list includes at least some Latin for all cardinals
    df["languages"] = [["Latin"] for _ in range(len(df))]
    
    # Add native language(s) based on country
    for i, cardinal in df.iterrows():
        country = cardinal["country"]
        if country in language_map:
            df.at[i, "languages"] = language_map[country]
        else:
            # Default to English + Latin for unlisted countries
            df.at[i, "languages"] = ["English", "Latin"]
        
        # Cardinals with Vatican roles likely speak Italian
        if "Vatican" in str(cardinal["current_role"]) and "Italian" not in df.at[i, "languages"]:
            df.at[i, "languages"].append("Italian")
            
        # Cardinals with international roles likely speak multiple languages
        if any(x in str(cardinal["current_role"]).lower() for x in ["international", "congregation", "secretary"]):
            if "English" not in df.at[i, "languages"]:
                df.at[i, "languages"].append("English")
            if "Italian" not in df.at[i, "languages"]:
                df.at[i, "languages"].append("Italian")
    
    # Convert to count for easy scoring
    df["language_count"] = df["languages"].apply(len)
    
    print(f"✅ Added language abilities estimates")
    return df

def calculate_diplomatic_experience(df):
    """Calculate diplomatic experience score based on roles and biography"""
    # Initialize with zeros
    df["diplomatic_score"] = 0.0
    
    # Diplomatic roles and keywords that indicate experience
    diplomatic_roles = [
        "secretary of state", "nuncio", "ambassador", "diplomatic", 
        "international", "relations", "foreign", "apostolic delegate"
    ]
    
    # Score based on role
    for i, cardinal in df.iterrows():
        role = str(cardinal["current_role"]).lower()
        
        # Secretary of State gets highest score
        if "secretary of state" in role:
            df.at[i, "diplomatic_score"] = 1.0
        else:
            # Add points for each diplomatic keyword found
            matches = sum(1 for term in diplomatic_roles if term in role)
            df.at[i, "diplomatic_score"] = min(matches * 0.25, 0.9)
    
    print(f"✅ Added diplomatic experience scores")
    return df

def calculate_pastoral_experience(df):
    """Calculate pastoral experience with diocesan leadership"""
    # Initialize with base score
    df["pastoral_score"] = 0.3  # Base pastoral experience for all cardinals
    
    # Pastoral roles and keywords
    pastoral_roles = [
        "bishop", "archbishop", "cardinal", "diocese", "archdiocese", 
        "pastoral", "parish", "clergy"
    ]
    
    # Score based on role
    for i, cardinal in df.iterrows():
        role = str(cardinal["current_role"]).lower()
        
        # Bishops and Archbishops get higher scores
        if "archbishop" in role:
            df.at[i, "pastoral_score"] = 0.9
        elif "bishop" in role:
            df.at[i, "pastoral_score"] = 0.8
        else:
            # Add points for each pastoral keyword found
            matches = sum(1 for term in pastoral_roles if term in role)
            df.at[i, "pastoral_score"] = min(0.3 + matches * 0.15, 0.9)
    
    print(f"✅ Added pastoral experience scores")
    return df

def calculate_vatican_score(df):
    """
    Calculate Vatican/Curia experience score based on roles.
    This is a separate function to ensure this score is calculated before coalition assignments.
    """
    # Initialize with zeros
    df["vatican_score"] = 0.0
    
    # Vatican/Curia roles and keywords
    vatican_roles = [
        "curia", "prefect", "dicastery", "congregation", 
        "secretary", "pontifical", "vatican", "holy see"
    ]
    
    # Score based on role
    for i, cardinal in df.iterrows():
        role = str(cardinal["current_role"]).lower()
        
        # Check for matches with Vatican roles
        matches = sum(1 for term in vatican_roles if term in role)
        df.at[i, "vatican_score"] = min(matches * 0.2, 0.9)
        
        # Give high score to specific high-ranking positions
        if any(x in role for x in ["prefect", "president", "secretary of state"]):
            df.at[i, "vatican_score"] = 0.9
    
    print(f"✅ Added Vatican experience scores")
    return df

def assign_coalition_probabilities(df):
    """
    Assign probabilities of belonging to various cardinal coalitions
    based on nationality, theology, and appointing pope
    """
    # Initialize coalition columns
    coalitions = [
        "european_bloc", "global_south_bloc", "italian_bloc",
        "conservative_bloc", "moderate_bloc", "progressive_bloc",
        "vatican_insiders", "outsiders"
    ]
    
    for coalition in coalitions:
        df[coalition] = 0.0
    
    # European bloc
    european_countries = ["Italy", "Germany", "France", "Spain", "Poland", "Austria", 
                          "Belgium", "Netherlands", "Switzerland", "Portugal", 
                          "Ireland", "United Kingdom"]
    df.loc[df["country"].isin(european_countries), "european_bloc"] = 0.8
    
    # Italian bloc
    df.loc[df["country"] == "Italy", "italian_bloc"] = 0.9
    df.loc[df["country"] != "Italy", "italian_bloc"] = 0.1
    
    # Global South bloc
    global_south = ["Brazil", "Argentina", "Mexico", "Peru", "Chile", "Colombia",
                    "Nigeria", "Kenya", "South Africa", "Ghana", "Congo",
                    "Philippines", "India", "Indonesia", "Vietnam"]
    df.loc[df["country"].isin(global_south), "global_south_bloc"] = 0.8
    
    # Theological blocs
    df.loc[df["theological_position"] <= 4.0, "conservative_bloc"] = 0.8
    df.loc[(df["theological_position"] > 4.0) & 
           (df["theological_position"] < 7.0), "moderate_bloc"] = 0.8
    df.loc[df["theological_position"] >= 7.0, "progressive_bloc"] = 0.8
    
    # Vatican insiders vs outsiders - using previously calculated vatican_score
    df.loc[df["vatican_score"] >= 0.7, "vatican_insiders"] = 0.9
    df.loc[df["vatican_score"] < 0.3, "outsiders"] = 0.8
    
    print(f"✅ Added coalition probabilities")
    return df

def calculate_media_presence(df):
    """Calculate media presence and public profile score"""
    # This would normally involve analysis of news mentions, social media, etc.
    # For demonstration, we'll use a simplified approach based on role
    
    df["media_score"] = 0.5  # Default middle score
    
    high_profile_roles = ["president", "prefect", "archbishop of", "major", "secretary of state"]
    
    for i, cardinal in df.iterrows():
        role = str(cardinal["current_role"]).lower()
        
        # Higher score for high-profile roles
        matches = sum(1 for term in high_profile_roles if term in role)
        if matches > 0:
            df.at[i, "media_score"] = min(0.5 + matches * 0.1, 0.9)
    
    print(f"✅ Added media presence scores")
    return df

def identify_previous_papabile(df):
    """
    Identify cardinals previously considered papabile
    In a real system, this would use historical data
    """
    # For demonstration, we'll mark a few random cardinals
    df["previous_papabile"] = False
    
    # Mark ~10% as previous papabile
    papabile_indices = np.random.choice(
        df.index, 
        size=int(len(df) * 0.1), 
        replace=False
    )
    df.loc[papabile_indices, "previous_papabile"] = True
    
    print(f"✅ Identified {sum(df['previous_papabile'])} previous papabili")
    return df

# === MAIN ENRICHMENT PIPELINE ===

def enrich_cardinal_data(input_csv="data/cardinals_raw.csv", output_csv="data/cardinals_enriched.csv"):
    """Main function to enrich cardinal data with additional predictive factors"""
    print(f"🔄 Starting cardinal data enrichment process...")
    
    # Load data
    df = load_raw_data(input_csv)
    
    # Apply enrichment functions - NOTE THE ORDER MATTERS!
    df = estimate_theological_position(df)
    df = estimate_language_abilities(df)
    df = calculate_diplomatic_experience(df)
    df = calculate_pastoral_experience(df)
    
    # Calculate Vatican score BEFORE coalition assignment
    df = calculate_vatican_score(df)
    
    # Now assign coalitions (which depend on vatican_score)
    df = assign_coalition_probabilities(df)
    df = calculate_media_presence(df)
    df = identify_previous_papabile(df)
    
    # Save enriched data
    df.to_csv(output_csv, index=False)
    print(f"✅ Enriched data saved to {output_csv}")
    
    return df

# === ENTRY POINT ===

if __name__ == "__main__":
    try:
        enriched_df = enrich_cardinal_data()
        print(f"✨ Cardinal data enrichment complete! Added {enriched_df.shape[1]} attributes to {enriched_df.shape[0]} cardinals.")
    except Exception as e:
        print(f"❌ Error during enrichment process: {e}")
        import traceback
        traceback.print_exc()
