from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json

# Import from models directory
from models.cardinals import load_cardinals, get_top_candidates

app = Flask(__name__)

# Global constants
DATA_PATH = "data/cardinals_scored.csv"

@app.route('/')
def index():
    """Main dashboard page"""
    # Load cardinals and get basic stats
    cardinals = load_cardinals(DATA_PATH)
    electors = cardinals[cardinals["age"] < 80]
    required_votes = int(np.ceil(2/3 * len(electors)))
    
    # Get top candidates
    top_cardinals = get_top_candidates(electors, top_n=5)
    
    # Stats to display
    stats = {
        "total_cardinals": len(cardinals),
        "eligible_electors": len(electors),
        "required_votes": required_votes,
        "italian_cardinals": sum(electors["country"] == "Italy"),
        "european_cardinals": sum(electors["country"].isin(["Italy", "Germany", "France", "Spain", "Poland"])),
        "global_south": sum(electors["country"].isin(["Brazil", "Mexico", "Argentina", "Nigeria", "South Africa", "Philippines", "India"]))
    }
    
    return render_template('index.html', stats=stats, top_cardinals=top_cardinals)

@app.route('/leaderboard')
def leaderboard():
    """Cardinal leaderboard page"""
    # Load cardinals
    cardinals = load_cardinals(DATA_PATH)
    electors = cardinals[cardinals["age"] < 80]
    
    # Get filter params from request
    min_age = request.args.get('min_age', default=0, type=int)
    max_age = request.args.get('max_age', default=100, type=int)
    country = request.args.get('country', default=None, type=str)
    sort_by = request.args.get('sort_by', default="ppi", type=str)
    
    # Apply filters
    filtered = electors[(electors["age"] >= min_age) & (electors["age"] <= max_age)]
    if country:
        filtered = filtered[filtered["country"] == country]
    
    # Sort data
    if sort_by == "ppi":
        score_col = "enhanced_ppi" if "enhanced_ppi" in filtered.columns else "ppi_score"
        sorted_data = filtered.sort_values(score_col, ascending=False)
    elif sort_by == "pvs":
        sorted_data = filtered.sort_values("pvs_score", ascending=False)
    elif sort_by == "age":
        sorted_data = filtered.sort_values("age")
    else:
        sorted_data = filtered.sort_values("country")
    
    # Get unique countries for filter dropdown
    countries = sorted(electors["country"].dropna().unique().tolist())
    
    return render_template('leaderboard.html', 
                           cardinals=sorted_data, 
                           countries=countries, 
                           min_age=min_age, 
                           max_age=max_age, 
                           selected_country=country,
                           sort_by=sort_by)

@app.route('/simulation')
def simulation():
    """Simple simulation page (no actual simulation yet)"""
    return render_template('simulation.html')

if __name__ == '__main__':
    # Run without debug mode to avoid watchdog issue
    app.run(host='127.0.0.1', port=5000)
