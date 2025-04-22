import pandas as pd
import numpy as np
import json
import random
import datetime
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ------------------------------------------------------------
# SECTION 1: Load and prepare eligible cardinal data
# ------------------------------------------------------------

def load_eligible_cardinals(csv_path="data/cardinals_scored.csv"):
    """
    Load cardinal data and return electors under age 80 with normalized weights.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback to raw data if scored data doesn't exist
        print(f"⚠️ Scored data not found at {csv_path}, falling back to raw data")
        df = pd.read_csv("data/cardinals_raw.csv")
        print(f"✅ Loaded {len(df)} cardinals from raw data as fallback")
        
        # Add basic scoring if not present
        if "ppi_score" not in df.columns:
            # Add simple age scoring
            df["age_score"] = df["age"].apply(lambda x: 1.0 if 65 <= x <= 78 else 0.5)
            # Add simple region scoring
            df["region_score"] = 0.5
            # Add simple vatican scoring based on role
            df["vatican_score"] = df["current_role"].apply(
                lambda x: 0.8 if any(k in str(x).lower() for k in ["curia", "prefect", "vatican"]) else 0.4
            )
            # Simple PPI
            df["ppi_score"] = 0.4 * df["age_score"] + 0.4 * df["vatican_score"] + 0.2 * df["region_score"]

    # Only cardinals under age 80 can vote and be elected
    electors = df[df["age"] < 80].copy()

    if electors.empty:
        raise ValueError("No eligible electors found (all are over 80).")
        
    # Use enhanced PPI as primary weight if available
    if "enhanced_ppi" in electors.columns:
        electors["base_weight"] = electors["enhanced_ppi"]
    elif "ppi_score" in electors.columns:
        electors["base_weight"] = electors["ppi_score"]
    else:
        # If no scoring available, use uniform weights
        electors["base_weight"] = 1.0
    
    # Add a very small random component to break ties
    electors["base_weight"] += np.random.random(len(electors)) * 0.01
    
    # Normalize weights
    total_weight = electors["base_weight"].sum()
    if total_weight > 0:
        electors["weight"] = electors["base_weight"] / total_weight
    else:
        # Fallback to uniform weights if sum is zero
        electors["weight"] = 1.0 / len(electors)
    
    print(f"✅ Loaded {len(electors)} eligible cardinal electors")
    return electors


# ------------------------------------------------------------
# SECTION 2: Basic utility function to simulate one voting round
# ------------------------------------------------------------

def simulate_ballot(electors):
    """
    Simulate one round of voting. Each elector casts one vote.
    """
    if "weight" not in electors.columns:
        # If weights weren't properly calculated, use uniform weights
        weights = np.ones(len(electors)) / len(electors)
    else:
        weights = electors["weight"].values
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    votes = np.random.choice(
        electors["name"],
        size=len(electors),
        p=weights
    )
    return pd.Series(votes).value_counts()


# ------------------------------------------------------------
# SECTION 3: Voting Dynamics & Coalition Functions
# ------------------------------------------------------------

def build_cardinal_coalitions(electors):
    """Model the informal coalitions among cardinals."""
    coalitions = {}
    
    # Check which coalition data is available
    coalition_cols = [col for col in electors.columns if col.endswith('_bloc') or col in ['vatican_insiders', 'outsiders']]
    
    if coalition_cols:
        # Use pre-calculated coalition probabilities if available
        for col in coalition_cols:
            coalition_name = col.replace('_bloc', '')
            coalitions[coalition_name] = electors[electors[col] > 0.6].index.tolist()
    else:
        # Fallback to geography-based coalitions
        european_countries = ["Italy", "Germany", "France", "Spain", "Poland", "Portugal", "Austria"]
        global_south_countries = ["Brazil", "Mexico", "Argentina", "Nigeria", "South Africa", "Philippines", "India"]
        italian_cardinals = electors[electors["country"] == "Italy"].index.tolist()
        
        european_bloc = electors[electors["country"].isin(european_countries)].index.tolist()
        global_south_bloc = electors[electors["country"].isin(global_south_countries)].index.tolist()
        
        # Create basic coalitions based on Vatican experience if available
        if "vatican_score" in electors.columns:
            vatican_insiders = electors[electors["vatican_score"] > 0.7].index.tolist()
            outsiders = electors[electors["vatican_score"] < 0.3].index.tolist()
        else:
            # Simple approximation based on role
            vatican_insiders = electors[electors["current_role"].str.contains("Vatican|Curia|Prefect", case=False, na=False)].index.tolist()
            outsiders = [i for i in electors.index if i not in vatican_insiders]
        
        coalitions = {
            "european": european_bloc,
            "global_south": global_south_bloc,
            "italian": italian_cardinals,
            "vatican_insiders": vatican_insiders,
            "outsiders": outsiders
        }
        
        # Try to infer theological positions if available
        if "theological_position" in electors.columns:
            coalitions["conservative"] = electors[electors["theological_position"] <= 4.0].index.tolist()
            coalitions["moderate"] = electors[(electors["theological_position"] > 4.0) & 
                                             (electors["theological_position"] < 7.0)].index.tolist()
            coalitions["progressive"] = electors[electors["theological_position"] >= 7.0].index.tolist()
    
    # Make sure we have at least something in each coalition
    for key in coalitions:
        if not coalitions[key]:
            # Assign random cardinals to empty coalitions
            coalitions[key] = np.random.choice(electors.index, size=max(int(len(electors)*0.1), 1), replace=False).tolist()
            
    print(f"✅ Built {len(coalitions)} cardinal coalitions")
    return coalitions


def calculate_coalition_support(candidate_name, electors, coalitions, voting_history=None):
    """
    Calculate coalition support for a candidate based on their faction memberships.
    
    Returns a dictionary of coalition support scores (0-1 scale).
    """
    support = {}
    candidate_idx = electors[electors["name"] == candidate_name].index
    
    if len(candidate_idx) == 0:
        # Candidate not found
        return {}
    
    # Get candidate's coalition memberships
    candidate_coalitions = []
    for coalition_name, members in coalitions.items():
        if candidate_idx[0] in members:
            candidate_coalitions.append(coalition_name)
    
    # Calculate support from each coalition
    for coalition_name, members in coalitions.items():
        # Base support if member of coalition
        if coalition_name in candidate_coalitions:
            base_support = 0.7  # Strong support from own coalition
        else:
            # Support from other coalitions based on alignment
            if coalition_name == "european" and "italian" in candidate_coalitions:
                base_support = 0.6  # Italians get good support from Europeans
            elif coalition_name == "moderate":
                # Moderates can support both conservatives and progressives somewhat
                if "conservative" in candidate_coalitions or "progressive" in candidate_coalitions:
                    base_support = 0.4
                else:
                    base_support = 0.3
            else:
                base_support = 0.2  # Default low support from other coalitions
        
        # Adjust for previous voting patterns if available
        if voting_history and len(voting_history) > 1:
            last_ballot = voting_history[-1]
            if candidate_name in last_ballot.index:
                # Count coalition member votes for this candidate in last ballot
                coalition_votes = 0
                for member_idx in members:
                    member_name = electors.loc[member_idx, "name"]
                    # This is a simplification - in reality we don't know who voted for whom
                    if random.random() < 0.5:  # 50% chance to count vote from coalition member
                        coalition_votes += 1
                
                # Adjust support based on previous votes
                vote_boost = min(coalition_votes / len(members), 0.3)
                base_support += vote_boost
        
        support[coalition_name] = min(base_support, 1.0)
    
    return support


def calculate_ballot_dynamics(day, ballot_num, voting_history, coalitions):
    """
    Calculate the dynamics that will influence the current ballot.
    Returns a dictionary of factors that adjust voting patterns.
    """
    dynamics = {
        "scatter_factor": 0.0,         # How scattered the votes are (high = more scattered)
        "momentum_factor": 0.0,        # How much previous frontrunners gain advantage
        "strategic_factor": 0.0,       # How much strategic voting occurs (vs. conscience)
        "coalition_strength": 1.0,     # How much coalitions vote as blocks vs. individually
        "frontrunner_resistance": 0.0  # Resistance to voting for the frontrunner
    }
    
    # Early ballots have more scattered "courtesy votes"
    if day == 1 and ballot_num == 1:
        dynamics["scatter_factor"] = 0.8  # Very scattered on first ballot
        dynamics["strategic_factor"] = 0.1  # Very little strategic voting
    elif day == 1:
        dynamics["scatter_factor"] = 0.6  # Still scattered on day 1
        dynamics["strategic_factor"] = 0.2
    elif day == 2:
        dynamics["scatter_factor"] = 0.4  # Getting more focused
        dynamics["strategic_factor"] = 0.4
    else:
        dynamics["scatter_factor"] = max(0.8 - (day * 0.15), 0.1)  # Gradually decreasing scatter
        dynamics["strategic_factor"] = min(0.3 + (day * 0.1), 0.9)  # Increasing strategic voting
    
    # Momentum builds for frontrunners as conclave progresses
    total_ballots = len(voting_history) if voting_history else 0
    if total_ballots > 0:
        dynamics["momentum_factor"] = min(total_ballots / 10, 0.6)
    
    # Coalition voting becomes stronger in later ballots
    dynamics["coalition_strength"] = min(1.0, 0.5 + (day * 0.1))
    
    # Frontrunner resistance can emerge in middle days if no consensus
    if day >= 3 and total_ballots >= 8:
        # If the same cardinal leads but can't reach 2/3, resistance emerges
        if total_ballots >= 2:
            prev_leaders = [ballot.index[0] if not ballot.empty else None for ballot in voting_history[-2:]]
            if prev_leaders[0] == prev_leaders[1] and prev_leaders[0] is not None:
                # Same leader for two ballots but not winning - resistance builds
                dynamics["frontrunner_resistance"] = min(0.3 + ((day - 3) * 0.1), 0.6)
    
    return dynamics


def simulate_ballot_with_dynamics(electors, coalitions, voting_history=None, dynamics=None):
    """
    Simulate one ballot with realistic conclave dynamics.
    
    Args:
        electors: DataFrame of cardinal electors
        coalitions: Dictionary of cardinal coalitions
        voting_history: List of previous ballot results
        dynamics: Dictionary of voting dynamics factors
    
    Returns:
        Series with vote counts for each candidate
    """
    if dynamics is None:
        # Default dynamics
        dynamics = {
            "scatter_factor": 0.5,
            "momentum_factor": 0.0,
            "strategic_factor": 0.2,
            "coalition_strength": 0.5,
            "frontrunner_resistance": 0.0
        }
    
    # Start with base weights
    ballot_weights = electors["weight"].copy()
    
    # Apply momentum to frontrunners from previous ballots
    if voting_history and len(voting_history) > 0 and dynamics["momentum_factor"] > 0:
        last_ballot = voting_history[-1]
        if not last_ballot.empty:
            frontrunners = last_ballot.nlargest(min(3, len(last_ballot))).index.tolist()
            
            # Create mapping from names to indices
            name_to_idx = {name: i for i, name in enumerate(electors["name"])}
            
            # Apply momentum boost to frontrunners
            for i, name in enumerate(frontrunners):
                if name in name_to_idx:
                    idx = name_to_idx[name]
                    position_factor = (3 - i) / 3  # 1.0 for first, 0.67 for second, 0.33 for third
                    momentum_boost = dynamics["momentum_factor"] * position_factor
                    
                    # Reduce boost if frontrunner resistance is high
                    if i == 0 and dynamics["frontrunner_resistance"] > 0:
                        momentum_boost *= (1 - dynamics["frontrunner_resistance"])
                    
                    ballot_weights.iloc[idx] *= (1 + momentum_boost)
    
    # Apply coalition voting patterns
    if dynamics["coalition_strength"] > 0:
        # Get coalition support for each candidate
        all_support = {}
        for name in electors["name"]:
            all_support[name] = calculate_coalition_support(name, electors, coalitions, voting_history)
        
        # Adjust weights based on coalition support
        for i, (_, cardinal) in enumerate(electors.iterrows()):
            name = cardinal["name"]
            
            # Skip if no coalition support data
            if name not in all_support or not all_support[name]:
                continue
            
            # Calculate average coalition support
            supports = all_support[name].values()
            if supports:  # Check if supports is not empty
                avg_support = sum(supports) / len(supports)
                
                # Apply coalition effect
                coalition_effect = dynamics["coalition_strength"] * (avg_support - 0.5) * 2
                ballot_weights.iloc[i] *= (1 + coalition_effect)
    
    # Apply scatter factor (randomness)
    if dynamics["scatter_factor"] > 0:
        random_factor = np.random.random(len(ballot_weights)) * dynamics["scatter_factor"]
        ballot_weights = ballot_weights * (1 - dynamics["scatter_factor"]) + random_factor
    
    # Normalize weights to ensure valid probability distribution
    # Add safety check to prevent division by zero
    if ballot_weights.sum() > 0:
        ballot_weights = ballot_weights / ballot_weights.sum()
    else:
        # If all weights are zero, assign uniform weights
        ballot_weights = pd.Series([1/len(electors)] * len(electors), index=ballot_weights.index)
    
    # Simulate the ballot by sampling according to weights
    votes = np.random.choice(
        electors["name"],
        size=len(electors),
        p=ballot_weights
    )
    
    return pd.Series(votes).value_counts()


def update_after_reflection(electors, voting_history):
    """
    Update cardinal preferences after a day of prayer and reflection.
    This can sometimes create unexpected shifts in voting patterns.
    """
    # Copy to avoid modifying original
    updated_electors = electors.copy()
    
    # Slight reset of momentum - prayer can cause cardinals to reconsider
    if "base_weight" in updated_electors.columns:
        # Blend current weight with original base weight
        updated_electors["weight"] = (
            updated_electors["weight"] * 0.7 + 
            updated_electors["base_weight"] * 0.3
        )
    
    # Small random adjustment to all weights
    random_adjustment = (np.random.random(len(updated_electors)) * 0.2) + 0.9  # 0.9-1.1 range
    updated_electors["weight"] = updated_electors["weight"] * random_adjustment
    
    # Normalize weights
    weight_sum = updated_electors["weight"].sum()
    if weight_sum > 0:
        updated_electors["weight"] = updated_electors["weight"] / weight_sum
    else:
        # Fallback to uniform weights if sum is zero
        updated_electors["weight"] = 1.0 / len(updated_electors)
    
    # There's a small chance a new candidate emerges after reflection
    if len(voting_history) > 5 and random.random() < 0.3:
        # Find cardinals who haven't received many votes
        all_votes = pd.concat(voting_history, axis=1).sum(axis=1)
        low_vote_cardinals = all_votes[all_votes < 3].index.tolist()
        
        if low_vote_cardinals:
            # Select one to emerge as a surprise candidate
            surprise_candidate = random.choice(low_vote_cardinals)
            surprise_idx = updated_electors[updated_electors["name"] == surprise_candidate].index
            
            if len(surprise_idx) > 0:
                # Boost their weight significantly
                idx = surprise_idx[0]
                updated_electors.at[idx, "weight"] = updated_electors.at[idx, "weight"] * 3
                
                # Normalize weights again
                updated_electors["weight"] = updated_electors["weight"] / updated_electors["weight"].sum()
    
    return updated_electors


def simulate_runoff(electors, top_two_candidates):
    """
    Simulate a runoff between the top two candidates.
    This happens after 33 ballots (~10 days) with no winner.
    """
    # Calculate support for each of the top two
    support = {}
    
    for candidate in top_two_candidates:
        # Base support from own votes
        candidate_idx = electors[electors["name"] == candidate].index
        if len(candidate_idx) == 0:
            support[candidate] = 0
            continue
        
        # Start with candidate's own weight
        candidate_weight = electors.loc[candidate_idx[0], "weight"]
        
        # Add support from other cardinals based on alignment
        for _, cardinal in electors.iterrows():
            if cardinal["name"] == candidate:
                continue  # Skip the candidate themselves
                
            # Factors that influence runoff support
            alignment_factors = []
            
            # Same region/country factor
            if "country" in electors.columns:
                if cardinal["country"] == electors.loc[candidate_idx[0], "country"]:
                    alignment_factors.append(0.7)
                
            # Theological alignment factor
            if "theological_position" in electors.columns:
                candidate_theology = electors.loc[candidate_idx[0], "theological_position"]
                cardinal_theology = cardinal["theological_position"]
                theology_diff = abs(candidate_theology - cardinal_theology)
                
                if theology_diff < 1:
                    alignment_factors.append(0.9)  # Very close alignment
                elif theology_diff < 2:
                    alignment_factors.append(0.7)  # Good alignment
                elif theology_diff < 3:
                    alignment_factors.append(0.5)  # Moderate alignment
                else:
                    alignment_factors.append(0.3)  # Poor alignment
            
            # Use the average of alignment factors as support probability
            if alignment_factors:
                support_prob = sum(alignment_factors) / len(alignment_factors)
                if random.random() < support_prob:
                    candidate_weight += cardinal["weight"]
        
        support[candidate] = candidate_weight
    
    # Normalize support
    total_support = sum(support.values())
    if total_support > 0:
        for candidate in support:
            support[candidate] /= total_support
    else:
        # Equal support if total is zero
        for candidate in support:
            support[candidate] = 1.0 / len(support)
    
    # Determine runoff winner
    vote_counts = {
        candidate: int(round(support[candidate] * len(electors)))
        for candidate in support
    }
    
    # Ensure we don't have more votes than electors
    total_votes = sum(vote_counts.values())
    if total_votes > len(electors):
        # Scale down proportionally
        for candidate in vote_counts:
            vote_counts[candidate] = int(vote_counts[candidate] * len(electors) / total_votes)
    
    # Return as a Series for consistency
    return pd.Series(vote_counts)


# ------------------------------------------------------------
# SECTION 4: Full Conclave Simulation
# ------------------------------------------------------------

def run_conclave(electors, max_rounds=10, verbose=True):
    """
    Simulate a papal conclave with basic voting.
    Returns the name of the elected pope, or None if no election.
    
    LEGACY FUNCTION - kept for backwards compatibility
    """
    required_votes = int(np.ceil(2/3 * len(electors)))
    
    if verbose:
        print(f"\n🕊️  Conclave begins with {len(electors)} electors.")
        print(f"⚖️  Required votes to win: {required_votes} (2/3 majority)\n")

    for round_num in range(1, max_rounds + 1):
        if verbose:
            print(f"🗳️  Round {round_num}")

        ballot = simulate_ballot(electors)

        top = ballot.head(3)
        for name, count in top.items():
            if verbose:
                print(f"   - {name}: {count} votes")

        if ballot.iloc[0] >= required_votes:
            winner = ballot.index[0]
            if verbose:
                print(f"\n🎉 {winner} is elected pope with {ballot.iloc[0]} votes!\n")
            return winner

        if verbose:
            print("   ↪️  No winner. Proceeding to next round...\n")

    if verbose:
        print("❌ No pope elected after max rounds.\n")
    return None


def run_realistic_conclave(electors, max_days=15, verbose=True):
    """
    Simulate a papal conclave with realistic timing rules and dynamics.
    
    Args:
        electors: DataFrame of cardinal electors
        max_days: Maximum number of days for the conclave
        verbose: Whether to print progress
    
    Returns:
        Tuple of (elected pope name, voting history, conclave stats)
    """
    required_votes = int(np.ceil(2/3 * len(electors)))
    voting_history = []
    conclave_stats = {
        "required_votes": required_votes,
        "total_electors": len(electors),
        "days": 0,
        "total_ballots": 0,
        "winner": None,
        "winning_ballot": None,
        "winning_day": None,
        "vote_shares": [],
        "top_candidates_by_day": []
    }
    
    # Build cardinal coalitions
    coalitions = build_cardinal_coalitions(electors)
    
    if verbose:
        print(f"\n🕊️  Conclave begins with {len(electors)} electors")
        print(f"⚖️  Required votes to win: {required_votes} (2/3 majority)\n")
    
    for day in range(1, max_days + 1):
        conclave_stats["days"] = day
        day_top_candidates = []
        
        if verbose:
            print(f"📅 Day {day} of the Conclave")
        
        # Number of ballots per day (1 on first day, 4 on subsequent days)
        ballots_today = 1 if day == 1 else 4
        
        for ballot_num in range(1, ballots_today + 1):
            conclave_stats["total_ballots"] += 1
            
            if verbose:
                print(f"  🗳️ Ballot {ballot_num} of Day {day} (#{conclave_stats['total_ballots']} overall)")
            
            # Calculate ballot dynamics for this round
            dynamics = calculate_ballot_dynamics(day, ballot_num, voting_history, coalitions)
            
            # Simulate this ballot with appropriate dynamics
            votes = simulate_ballot_with_dynamics(electors, coalitions, voting_history, dynamics)
            voting_history.append(votes)
            
            # Record vote shares for analysis
            top_votes = votes.iloc[0] if not votes.empty else 0
            vote_share = top_votes / len(electors)
            conclave_stats["vote_shares"].append(vote_share)
            
            # Show top candidates
            top_candidates = votes.head(3)
            day_top_candidates.append(top_candidates.index.tolist()[:min(3, len(top_candidates))])
            
            if verbose:
                for name, count in top_candidates.items():
                    share = count / len(electors) * 100
                    print(f"    - {name}: {count} votes ({share:.1f}%)")
            
            # Check if we have a winner
            if not top_candidates.empty and top_candidates.iloc[0] >= required_votes:
                winner = top_candidates.index[0]
                conclave_stats["winner"] = winner
                conclave_stats["winning_ballot"] = conclave_stats["total_ballots"]
                conclave_stats["winning_day"] = day
                
                if verbose:
                    share = top_candidates.iloc[0] / len(electors) * 100
                    print(f"\n✨ {winner} elected pope with {top_candidates.iloc[0]} votes ({share:.1f}%)")
                    print(f"   Election occurred on Day {day}, Ballot {ballot_num} (#{conclave_stats['total_ballots']} overall)\n")
                
                return winner, voting_history, conclave_stats
            
            if verbose:
                print("    ↪️  No winner. Proceeding...")
        
        # Record top candidates from this day
        conclave_stats["top_candidates_by_day"].append(day_top_candidates)
        
        # Handle special conclave rules
        if day % 3 == 0 and day < 10:
            if verbose:
                print("\n  🙏 Day of prayer and reflection")
            # Reset some voting dynamics after prayer day
            electors = update_after_reflection(electors, voting_history)
        
        # Implement Benedict XVI's rule: after 33 ballots (≈10 days), go to runoff
        if day == 10:
            if verbose:
                print("\n  ⚡ Entering runoff between top two candidates")
            
            # Identify top two candidates from last ballot
            if not voting_history[-1].empty:
                top_two = voting_history[-1].nlargest(min(2, len(voting_history[-1]))).index.tolist()
                
                if verbose:
                    print(f"    Runoff between: {' and '.join(top_two)}")
                
                # Simulate runoff
                runoff_result = simulate_runoff(electors, top_two)
                voting_history.append(runoff_result)
                conclave_stats["total_ballots"] += 1
                
                # Declare winner
                winner = runoff_result.idxmax()
                winning_votes = runoff_result.max()
                
                conclave_stats["winner"] = winner
                conclave_stats["winning_ballot"] = conclave_stats["total_ballots"]
                conclave_stats["winning_day"] = day
                
                if verbose:
                    share = winning_votes / len(electors) * 100
                    print(f"\n✨ {winner} elected pope in runoff with {winning_votes} votes ({share:.1f}%)")
                    print(f"   Election occurred on Day {day} in final runoff (ballot #{conclave_stats['total_ballots']})\n")
                
                return winner, voting_history, conclave_stats
    
    if verbose:
        print("❌ No pope elected after maximum days.\n")
    
    return None, voting_history, conclave_stats


# ------------------------------------------------------------
# SECTION 5: Monte Carlo Simulation for Prediction
# ------------------------------------------------------------

def run_monte_carlo_simulation(electors, num_simulations=1000, max_days=15, verbose=True):
    """
    Run multiple conclave simulations to generate election probabilities.
    
    Args:
        electors: DataFrame of cardinal electors
        num_simulations: Number of simulations to run
        max_days: Maximum days for each conclave
        verbose: Whether to print progress
    
    Returns:
        Dictionary with simulation results and statistics
    """
    if verbose:
        print(f"🔄 Running {num_simulations} conclave simulations...")
    
    winners = []
    days_to_election = []
    ballots_to_election = []
    vote_shares = []
    
    for i in range(num_simulations):
        if verbose and (i+1) % 100 == 0:
            print(f"  Simulation {i+1}/{num_simulations} ({(i+1)/num_simulations*100:.1f}%)")
        
        winner, voting_history, stats = run_realistic_conclave(electors.copy(), max_days, verbose=False)
        
        winners.append(winner)
        if winner is not None:
            days_to_election.append(stats["winning_day"])
            ballots_to_election.append(stats["winning_ballot"])
            
            # Calculate final vote share
            final_ballot = voting_history[-1]
            winning_votes = final_ballot.loc[winner]
            vote_shares.append(winning_votes / len(electors))
    
    # Calculate winner probabilities
    winner_counts = pd.Series(winners).value_counts()
    winner_probs = winner_counts / num_simulations
    
    # Calculate statistics
    simulation_stats = {
        "most_likely_pope": winner_probs.index[0] if not winner_probs.empty else None,
        "election_probability": winner_probs.iloc[0] if not winner_probs.empty else 0,
        "top_candidates": winner_probs.head(10).to_dict(),
        "median_days": np.median(days_to_election) if days_to_election else None,
        "median_ballots": np.median(ballots_to_election) if ballots_to_election else None,
        "average_vote_share": np.mean(vote_shares) if vote_shares else None,
        "no_election_probability": winner_counts.get(None, 0) / num_simulations
    }
    
    if verbose:
        print("\n✅ Monte Carlo simulation complete!")
        print(f"📊 Election Probabilities:")
        for name, prob in list(simulation_stats["top_candidates"].items())[:5]:
            if name is not None:
                print(f"  - {name}: {prob*100:.1f}%")
        
        if simulation_stats["no_election_probability"] > 0:
            print(f"  - No election: {simulation_stats['no_election_probability']*100:.1f}%")
            
        print(f"\n⏱️ Expected conclave duration: {simulation_stats['median_days']:.1f} days "
              f"({simulation_stats['median_ballots']:.1f} ballots)")
    
    return simulation_stats


# ------------------------------------------------------------
# SECTION 6: Main Entry Point
# ------------------------------------------------------------

def predict_pope(input_csv="data/cardinals_scored.csv", num_simulations=1000, verbose=True):
    """
    Run papal prediction algorithm based on cardinal data.
    
    Args:
        input_csv: Path to scored cardinal data
        num_simulations: Number of Monte Carlo simulations to run
        verbose: Whether to print detailed progress
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Make sure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Load eligible cardinals
        electors = load_eligible_cardinals(input_csv)
        
        if verbose:
            print(f"\n🔮 PAPAL CONCLAVE PREDICTION")
            print(f"===========================")
            print(f"Input data: {input_csv}")
            print(f"Eligible electors: {len(electors)}")
            print(f"Running {num_simulations} simulations...")
        
        # Run a single sample conclave to show dynamics
        if verbose:
            print("\n📋 SAMPLE CONCLAVE SIMULATION")
            print("============================")
            winner, voting_history, stats = run_realistic_conclave(electors.copy(), verbose=verbose)
            print(f"Sample result: {winner} elected on day {stats['winning_day']}, ballot {stats['winning_ballot']}")
        
        # Run Monte Carlo simulation - use fewer simulations if verbose for speed
        sim_count = min(num_simulations, 100) if verbose else num_simulations
        
        if verbose:
            print("\n📊 MONTE CARLO ANALYSIS")
            print("=====================")
        
        simulation_stats = run_monte_carlo_simulation(
            electors, 
            num_simulations=sim_count, 
            verbose=verbose
        )
        
        # Prepare final prediction results
        prediction = {
            "most_likely_pope": simulation_stats["most_likely_pope"],
            "probability": simulation_stats["election_probability"],
            "top_candidates": simulation_stats["top_candidates"],
            "expected_conclave_days": simulation_stats["median_days"],
            "expected_conclave_ballots": simulation_stats["median_ballots"],
            "expected_vote_share": simulation_stats["average_vote_share"],
            "no_election_probability": simulation_stats["no_election_probability"],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "electors_count": len(electors)
        }
        
        if verbose:
            print(f"\n🏆 PREDICTION RESULT")
            print(f"==================")
            print(f"Most likely pope: {prediction['most_likely_pope']} ({prediction['probability']*100:.1f}%)")
            print(f"Expected conclave duration: {prediction['expected_conclave_days']:.1f} days")
            print(f"Expected winning vote share: {prediction['expected_vote_share']*100:.1f}%")
        
        return prediction
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # Use a smaller number of simulations for faster testing
        prediction = predict_pope(num_simulations=10)
        print("\n✨ Habemus Papam! Prediction complete.")
    except Exception as e:
        print(f"⚠️ Error: {e}")
