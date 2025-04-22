import pandas as pd
import numpy as np
import os
from datetime import datetime

# Load enriched data
def load_enriched_data(csv_path="data/cardinals_enriched.csv"):
    """Load the enriched cardinal data with additional attributes."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} cardinals from enriched data")
    except FileNotFoundError:
        print(f"⚠️ Enriched data not found at {csv_path}, falling back to raw data")
        df = pd.read_csv("data/cardinals_raw.csv")
        print(f"✅ Loaded {len(df)} cardinals from raw data as fallback")
    return df

# === STEP 1: ENHANCED PPI SCORING ===

def compute_age_score(age):
    """
    Calculate age score based on optimal papal age range.
    Recent trends favor cardinals 65-78.
    """
    if pd.isna(age): return 0
    if 65 <= age <= 78: return 1.0
    if 60 <= age <= 82: return 0.7
    if 55 <= age <= 85: return 0.4
    return 0.1

def compute_region_score(country, regional_trend="global"):
    """
    Calculate region score based on current Church demographics.
    Parameter 'regional_trend' can adjust for historical trends:
      - "global": balanced global representation (default)
      - "growth": favor regions with Church growth (Global South)
      - "tradition": favor traditional Catholic regions (Europe)
    """
    if pd.isna(country): return 0.5
    
    # Regional classification
    europe = ["Italy", "Germany", "France", "Spain", "Poland", "Austria", "Portugal"]
    latin_america = ["Brazil", "Mexico", "Argentina", "Chile", "Colombia", "Peru"]
    north_america = ["United States", "Canada"]
    africa = ["Nigeria", "Ghana", "South Africa", "Congo", "Kenya"]
    asia = ["Philippines", "India", "Indonesia", "Vietnam", "South Korea"]
    
    # Score based on regional trend
    if regional_trend == "growth":
        # Favor regions with Church growth
        if country in africa: return 1.0
        if country in asia: return 0.9
        if country in latin_america: return 0.8
        if country in north_america: return 0.6
        if country in europe: 
            return 0.9 if country == "Italy" else 0.5  # Italy still has special status
    
    elif regional_trend == "tradition":
        # Favor traditional Catholic strongholds
        if country in europe: 
            return 1.0 if country == "Italy" else 0.9
        if country in latin_america: return 0.7
        if country in north_america: return 0.6
        if country in africa or country in asia: return 0.5
    
    else:  # "global" balanced approach
        if country in europe: 
            return 0.9 if country == "Italy" else 0.7
        if country in latin_america: return 0.8
        if country in africa: return 0.9
        if country in asia: return 0.8
        if country in north_america: return 0.7
    
    # Default for countries not in our lists
    return 0.6

def compute_vatican_score(role, diplomatic_score=0, pastoral_score=0):
    """
    Calculate Vatican/Curia experience score.
    Incorporates diplomatic and pastoral experience.
    """
    if pd.isna(role): return 0
    
    role = role.lower()
    vatican_keywords = [
        "curia", "prefect", "dicastery", "congregation", 
        "secretary", "pontifical", "vatican"
    ]
    
    # Base score from role keywords
    base_score = sum(0.1 for k in vatican_keywords if k in role)
    base_score = min(base_score, 1.0)
    
    # Incorporate diplomatic and pastoral experience
    composite_score = (base_score * 0.6) + (diplomatic_score * 0.3) + (pastoral_score * 0.1)
    
    return min(composite_score, 1.0)

def compute_theological_alignment_score(theological_position, context_trend=6.2):
    """
    Calculate how well a cardinal's theological position aligns
    with the current context of the Church.
    
    The context_trend parameter represents the current theological
    center of gravity in the College of Cardinals.
    Higher values = more progressive, lower = more conservative.
    """
    # Handle missing data
    if pd.isna(theological_position): return 0.5
    
    # Calculate distance from current context
    distance = abs(theological_position - context_trend)
    
    # Convert to a score (closer = higher score)
    if distance < 1: return 1.0
    if distance < 2: return 0.8
    if distance < 3: return 0.6
    if distance < 4: return 0.4
    return 0.2

def compute_language_score(language_count):
    """Calculate score based on languages spoken."""
    if pd.isna(language_count): return 0.5
    if language_count >= 5: return 1.0
    if language_count >= 4: return 0.9
    if language_count >= 3: return 0.7
    if language_count >= 2: return 0.5
    return 0.3

def compute_coalition_score(cardinal, coalition_weights=None):
    """
    Calculate how well a cardinal can unite various factions.
    Coalition weights can be adjusted to favor certain groups.
    """
    if coalition_weights is None:
        # Default coalition importance weights
        coalition_weights = {
            "european_bloc": 0.15,
            "global_south_bloc": 0.15,
            "italian_bloc": 0.1,
            "conservative_bloc": 0.15,
            "moderate_bloc": 0.2,
            "progressive_bloc": 0.15,
            "vatican_insiders": 0.05,
            "outsiders": 0.05
        }
    
    # Calculate weighted sum of coalition memberships
    coalition_score = 0
    for coalition, weight in coalition_weights.items():
        if coalition in cardinal and not pd.isna(cardinal[coalition]):
            coalition_score += cardinal[coalition] * weight
    
    return min(coalition_score, 1.0)

def compute_media_visibility_score(media_score, previous_papabile):
    """Calculate score based on media visibility and previous papabile status."""
    base_score = media_score if not pd.isna(media_score) else 0.5
    
    # Previous papabile status can help or hurt
    if previous_papabile:
        # Being previously considered papabile can be a mixed blessing
        # Sometimes it helps visibility, sometimes it creates resistance
        papabile_factor = np.random.choice([-0.1, 0.2])
        base_score += papabile_factor
    
    return min(max(base_score, 0), 1.0)

# === STEP 2: CALCULATE ENHANCED PPI ===

def calculate_enhanced_ppi(df, weights=None, context=None):
    """
    Calculate Enhanced Pope Potential Index with weighted components
    and contextual factors.
    """
    if weights is None:
        weights = {
            "age": 0.15,
            "region": 0.10,
            "vatican": 0.15,
            "theological": 0.20,
            "language": 0.10,
            "coalition": 0.20,
            "media": 0.10
        }
    
    if context is None:
        context = {
            "regional_trend": "global",
            "theological_trend": 6.2,
            "coalition_priorities": None
        }
    
    # Apply component scoring functions
    df["age_score"] = df["age"].apply(compute_age_score)
    df["region_score"] = df["country"].apply(lambda x: compute_region_score(x, context["regional_trend"]))
    
    # Check if we have diplomatic and pastoral scores or need to compute them
    if "diplomatic_score" not in df.columns:
        df["diplomatic_score"] = 0.0
    if "pastoral_score" not in df.columns:
        df["pastoral_score"] = 0.0
    
    # Check if vatican_score already exists (from enrichment) or compute it
    if "vatican_score" not in df.columns:
        # Compute Vatican score
        df["vatican_score"] = df.apply(
            lambda row: compute_vatican_score(
                row["current_role"], 
                row.get("diplomatic_score", 0), 
                row.get("pastoral_score", 0)
            ), 
            axis=1
        )
    
    # Additional component scores
    if "theological_position" in df.columns:
        df["theological_score"] = df["theological_position"].apply(
            lambda x: compute_theological_alignment_score(x, context["theological_trend"])
        )
    else:
        df["theological_score"] = 0.5  # Default if not available
    
    if "language_count" in df.columns:
        df["language_score"] = df["language_count"].apply(compute_language_score)
    else:
        df["language_score"] = 0.5  # Default if not available
    
    # Check if we have coalition data
    has_coalition_data = all(col in df.columns for col in [
        "european_bloc", "global_south_bloc", "italian_bloc",
        "conservative_bloc", "moderate_bloc", "progressive_bloc",
        "vatican_insiders", "outsiders"
    ])
    
    if has_coalition_data:
        df["coalition_score"] = df.apply(
            lambda row: compute_coalition_score(row, context["coalition_priorities"]), 
            axis=1
        )
    else:
        df["coalition_score"] = 0.5  # Default if not available
    
    # Media visibility
    if "media_score" in df.columns and "previous_papabile" in df.columns:
        df["visibility_score"] = df.apply(
            lambda row: compute_media_visibility_score(
                row.get("media_score", 0.5), 
                row.get("previous_papabile", False)
            ),
            axis=1
        )
    else:
        df["visibility_score"] = 0.5  # Default if not available
    
    # Calculate final PPI
    df["enhanced_ppi"] = (
        weights["age"] * df["age_score"] +
        weights["region"] * df["region_score"] +
        weights["vatican"] * df["vatican_score"] +
        weights["theological"] * df["theological_score"] +
        weights["language"] * df["language_score"] +
        weights["coalition"] * df["coalition_score"] +
        weights["media"] * df["visibility_score"]
    )
    
    print(f"✅ Calculated Enhanced PPI scores")
    return df

# === STEP 3: PVS SCORING (PAPAL VIBES SCORE) ===

def compute_name_legend_score(name):
    """
    Calculate name score based on historical papal name patterns.
    """
    if pd.isna(name): return 0
    
    # Historically popular papal names
    legendary_names = {
        "John": 1.0,      # 23 popes
        "Gregory": 0.9,   # 16 popes
        "Benedict": 0.9,  # 16 popes
        "Clement": 0.8,   # 14 popes
        "Innocent": 0.8,  # 13 popes
        "Leo": 0.8,       # 13 popes
        "Pius": 0.7,      # 12 popes
        "Stephen": 0.7,   # 9 popes
        "Urban": 0.7,     # 8 popes
        "Alexander": 0.6, # 8 popes
        "Francis": 0.6,   # Current pope
    }
    
    # Check for name matches
    for papal_name, score in legendary_names.items():
        if papal_name in name:
            return score
    
    return 0.5  # Default score for other names

def compute_charisma_score(age, media_score, languages):
    """Estimate charisma based on age, media presence, and languages."""
    # Baseline charisma
    base_score = 0.5
    
    # Age factor (slight preference for middle-aged)
    if 65 <= age <= 75:
        age_factor = 0.1
    elif 60 <= age <= 80:
        age_factor = 0.05
    else:
        age_factor = 0
    
    # Media presence factor
    media_factor = media_score * 0.2 if not pd.isna(media_score) else 0
    
    # Language versatility factor
    language_factor = min((languages - 1) * 0.05, 0.2) if not pd.isna(languages) else 0
    
    return min(base_score + age_factor + media_factor + language_factor, 1.0)

def compute_regional_symbolism_score(country):
    """
    Calculate the symbolic value of electing a pope from a particular region.
    """
    if pd.isna(country): return 0.5
    
    # Symbolic value of different regions
    symbolism_map = {
        # Global South (high symbolic value for growth regions)
        "Nigeria": 0.9, "Ghana": 0.9, "Philippines": 0.9, "Brazil": 0.9,
        "Mexico": 0.8, "Argentina": 0.8, "India": 0.9, "Vietnam": 0.8,
        
        # Traditional European Catholic countries (moderate symbolic value)
        "Italy": 0.6, "Poland": 0.7, "Spain": 0.6, "France": 0.6,
        
        # Other regions (moderate to low symbolic value)
        "United States": 0.5, "Canada": 0.5, "Germany": 0.6,
        "Australia": 0.5, "United Kingdom": 0.7
    }
    
    # Return symbolic value or default
    return symbolism_map.get(country, 0.5)

# === STEP 4: CALCULATE PVS ===

def calculate_pvs(df, weights=None):
    """
    Calculate Papal Vibes Score (PVS) with weighted components.
    """
    if weights is None:
        weights = {
            "name_legend": 0.25,
            "charisma": 0.35,
            "symbolism": 0.40
        }
    
    # Apply component scoring functions
    df["name_legend_score"] = df["name"].apply(compute_name_legend_score)
    
    # Calculate charisma with fallbacks for missing data
    df["charisma_score"] = df.apply(
        lambda row: compute_charisma_score(
            row["age"], 
            row.get("media_score", 0.5), 
            row.get("language_count", 2)
        ),
        axis=1
    )
    
    df["symbolism_score"] = df["country"].apply(compute_regional_symbolism_score)
    
    # Calculate final PVS
    df["pvs_score"] = (
        weights["name_legend"] * df["name_legend_score"] +
        weights["charisma"] * df["charisma_score"] +
        weights["symbolism"] * df["symbolism_score"]
    )
    
    print(f"✅ Calculated PVS scores")
    return df

# === MAIN FUNCTION ===

def score_cardinals(input_csv="data/cardinals_enriched.csv", output_csv="data/cardinals_scored.csv"):
    """Main function to calculate cardinal scores"""
    print(f"🔄 Starting cardinal scoring process...")
    
    # Load data
    df = load_enriched_data(input_csv)
    
    # Current context for PPI calculation
    context = {
        "regional_trend": "global",          # Balanced global approach
        "theological_trend": 6.2,            # Slightly progressive (reflecting Francis influence)
        "coalition_priorities": {
            "european_bloc": 0.15,
            "global_south_bloc": 0.20,       # Emphasis on Global South
            "italian_bloc": 0.05,
            "conservative_bloc": 0.15,
            "moderate_bloc": 0.25,           # Emphasis on moderates
            "progressive_bloc": 0.10,
            "vatican_insiders": 0.05,
            "outsiders": 0.05
        }
    }
    
    # Calculate Enhanced PPI
    df = calculate_enhanced_ppi(df, context=context)
    
    # Calculate PVS
    df = calculate_pvs(df)
    
    # Overall "Papabile Score" (combining both)
    df["papabile_score"] = (df["enhanced_ppi"] * 0.7) + (df["pvs_score"] * 0.3)
    
    # Save scored data
    df.to_csv(output_csv, index=False)
    print(f"✅ Scored data saved to {output_csv}")
    
    # Display top candidates
    top_candidates = df.sort_values("papabile_score", ascending=False).head(5)
    print("\n🔝 Top Papabili:")
    for i, (_, candidate) in enumerate(top_candidates.iterrows()):
        print(f"  {i+1}. {candidate['name']} ({candidate['country']}) - PPI: {candidate['enhanced_ppi']:.2f}, PVS: {candidate['pvs_score']:.2f}")
    
    return df

# === ENTRY POINT ===

if __name__ == "__main__":
    try:
        # Make sure data directory exists
        os.makedirs("data", exist_ok=True)
        
        scored_df = score_cardinals()
        print(f"\n✨ Cardinal scoring complete!")
    except Exception as e:
        print(f"❌ Error during scoring process: {e}")
        import traceback
        traceback.print_exc()
