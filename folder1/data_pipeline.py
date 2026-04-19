import json
import os
import glob
import pandas as pd

def build_historical_database(data_directory: str) -> pd.DataFrame:
    """Scrapes all JSON files and flattens them into a master DataFrame."""
    print("[SYSTEM] Igniting Data Parser...")
    
    all_deliveries = []
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(data_directory, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            match_data = json.load(f)
            
            # Extract high-level match info
            info = match_data.get("info", {})
            teams = info.get("teams", [])
            venue = info.get("venue", "Unknown")
            date = info.get("dates", ["Unknown"])[0]
            
            if "innings" not in match_data:
                continue
                
            # Dig into the innings, overs, and deliveries
            for innings in match_data["innings"]:
                batting_team = innings.get("team")
                bowling_team = teams[1] if teams[0] == batting_team else teams[0]
                
                for over in innings.get("overs", []):
                    over_num = over.get("over")
                    
                    for ball_idx, delivery in enumerate(over.get("deliveries", [])):
                        
                        runs_batter = delivery.get("runs", {}).get("batter", 0)
                        runs_extras = delivery.get("runs", {}).get("extras", 0)
                        is_wicket = 1 if "wickets" in delivery else 0
                        
                        # Flatten the delivery data
                        ball_data = {
                            "date": date,
                            "venue": venue,
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "over": over_num,
                            "ball_in_over": ball_idx + 1,
                            "batter": delivery.get("batter"),
                            "bowler": delivery.get("bowler"),
                            "runs": runs_batter + runs_extras,
                            "is_wicket": is_wicket
                        }
                        all_deliveries.append(ball_data)
                        
    # Convert the massive list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(all_deliveries)
    print(f"[SYSTEM] Parsed {len(df)} historical deliveries successfully.")
    return df

# Test the extraction
if __name__ == "__main__":
    # Ensure this path points to your unzipped folder
    master_dataframe = build_historical_database("ipl_json_data")
    print(master_dataframe.head())

     # --- ADD THESE TWO LINES ---
    print("[SYSTEM] Caching to CSV for rapid boot times...")
    master_dataframe.to_csv("historical_ipl.csv", index=False)
    print("[SYSTEM] Cache complete. Ready for Math Engine.")
