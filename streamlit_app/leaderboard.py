import pandas as pd
import os

def display_papal_leaderboard(csv_path="data/cardinals_scored.csv", top_n=20):
    """
    Display a leaderboard of top papal candidates based on scored data.
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: File {csv_path} not found.")
        return
    
    # Load cardinal data
    df = pd.read_csv(csv_path)
    
    # Filter to eligible electors (under 80)
    electors = df[df["age"] < 80].copy()
    
    # Determine which scoring columns exist
    score_col = "enhanced_ppi" if "enhanced_ppi" in electors.columns else "ppi_score"
    
    # Sort by primary score
    leaderboard = electors.sort_values(score_col, ascending=False)
    
    # Print leaderboard header
    print("\n" + "=" * 80)
    print(f"📊 POPECAST 2025 LEADERBOARD - TOP {top_n} CARDINALS")
    print("=" * 80)
    
    # Print column headers
    print(f"{'Rank':<5}{'Name':<30}{'Age':<5}{'Country':<15}{'PPI':<8}{'PVS':<8}")
    print("-" * 80)
    
    # Print top N cardinals
    for i, (_, cardinal) in enumerate(leaderboard.head(top_n).iterrows()):
        ppi = cardinal[score_col]
        pvs = cardinal.get("pvs_score", 0)
        
        print(f"{i+1:<5}{cardinal['name']:<30}{int(cardinal['age']):<5}{cardinal['country']:<15}{ppi:.3f}  {pvs:.3f}")
    
    print("=" * 80)
    print(f"Total eligible electors: {len(electors)}")
    print(f"Required votes to win: {int(len(electors) * 2/3)}")
    print("=" * 80)

if __name__ == "__main__":
    display_papal_leaderboard()
