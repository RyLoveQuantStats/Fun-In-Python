import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    confusion_matrix
)

import nfl_data_py as nfl

# Configure Logging
logging.basicConfig(
    filename='nfl_analysis.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Ethical Considerations Reminder
print("Reminder: Use any betting strategies developed responsibly and ethically. Ensure compliance with all relevant laws and regulations.")

# Step 1: Data Collection

def fetch_team_seasonal_data(years=[2023]):
    """
    Fetches player-level seasonal data for the specified years.
    """
    try:
        data = nfl.import_seasonal_data(years=years, s_type='REG')  # Specify season type if needed
        logging.info(f"Successfully fetched seasonal data for years: {years}")
        print(f"Successfully fetched seasonal data for years: {years}")
        return data
    except Exception as e:
        logging.error(f"Failed to fetch seasonal team data: {e}")
        print(f"Failed to fetch seasonal team data: {e}")
        raise

def fetch_pbp_data(years=[2023]):
    """
    Fetches play-by-play data for the specified years and aggregates it to game-level data.
    """
    try:
        print("Fetching Play-by-Play (PBP) data...")
        logging.info(f"Starting to fetch PBP data for years: {years}")
        
        # Fetch play-by-play data without specifying columns to ensure all necessary data is retrieved
        pbp_data = nfl.import_pbp_data(years=years, columns=None, downcast=True, cache=False, alt_path=None)
        
        logging.info(f"Successfully fetched PBP data for years: {years}")
        print(f"Successfully fetched PBP data for years: {years}")
        
        # Inspect the columns to identify necessary game-level information
        print("Columns in PBP data:", pbp_data.columns.tolist())
        logging.info(f"PBP data columns: {pbp_data.columns.tolist()}")
        
        # Identify if game-level columns are present
        required_pbp_cols = {'game_id', 'home_team', 'away_team', 'home_score', 'away_score'}
        missing_pbp_cols = required_pbp_cols - set(pbp_data.columns)
        if missing_pbp_cols:
            raise KeyError(f"PBP data is missing required columns: {missing_pbp_cols}")
        
        # Aggregate PBP data to game-level scores
        game_scores = pbp_data.groupby(['game_id', 'home_team', 'away_team']).agg(
            home_score=pd.NamedAgg(column='home_score', aggfunc='max'),
            away_score=pd.NamedAgg(column='away_score', aggfunc='max')
        ).reset_index()
        
        # Verify aggregation
        print("Sample aggregated game-level data:")
        print(game_scores.head())
        logging.info(f"Sample aggregated game-level data:\n{game_scores.head()}")
        
        return game_scores
    except Exception as e:
        logging.error(f"Failed to fetch and aggregate PBP data: {e}")
        print(f"Failed to fetch and aggregate PBP data: {e}")
        raise

def fetch_roster_data(years=[2023]):
    """
    Fetches seasonal roster data to map player IDs to their respective teams.
    """
    try:
        # Specify the columns you need; typically 'player_id' and 'team'
        columns = ['player_id', 'team']
        data = nfl.import_seasonal_rosters(years=years, columns=columns)
        logging.info(f"Successfully fetched roster data for years: {years}")
        print(f"Successfully fetched roster data for years: {years}")
        return data
    except Exception as e:
        logging.error(f"Failed to fetch roster data: {e}")
        print(f"Failed to fetch roster data: {e}")
        raise

# Step 2: Data Cleaning and Aggregation

def clean_team_seasonal_data(player_df, game_df, roster_df, handle_ties='half'):
    """
    Cleans and aggregates player-level data to team-level statistics.
    
    Parameters:
    - handle_ties: Strategy to handle tied games ('ignore', 'half', 'full').
    """
    logging.info("Starting data cleaning process...")
    try:
        # Merge player data with roster data to get team information
        if 'player_id' not in player_df.columns:
            raise KeyError("'player_id' column is missing from player data.")
        if 'player_id' not in roster_df.columns or 'team' not in roster_df.columns:
            raise KeyError("'player_id' or 'team' column is missing from roster data.")
        
        player_team_df = player_df.merge(
            roster_df[['player_id', 'team']],
            on='player_id',
            how='left'
        )
        
        # Check for any players without team mapping
        unmapped_players = player_team_df['team'].isnull().sum()
        if unmapped_players > 0:
            logging.warning(f"{unmapped_players} players could not be mapped to a team and will be excluded.")
            print(f"Warning: {unmapped_players} players could not be mapped to a team and will be excluded.")
            player_team_df = player_team_df.dropna(subset=['team'])
        
        # Aggregate player-level data to team-level statistics
        aggregation_functions = {
            'fantasy_points': 'sum',
            'completions': 'sum',
            'passing_yards': 'sum',
            'rushing_yards': 'sum',
            'receiving_yards': 'sum',
            'games': 'sum',
            # Add other relevant aggregations if needed
        }
        team_data = player_team_df.groupby('team', as_index=False).agg(aggregation_functions)
        
        # Feature Engineering: Compute additional metrics
        team_data['points'] = team_data['passing_yards'] + team_data['rushing_yards'] + team_data['receiving_yards']
        team_data['avg_fantasy_points'] = team_data['fantasy_points'] / team_data['games']
        team_data['avg_passing_yards'] = team_data['passing_yards'] / team_data['games']
        team_data['avg_rushing_yards'] = team_data['rushing_yards'] / team_data['games']
        team_data['avg_receiving_yards'] = team_data['receiving_yards'] / team_data['games']
        
        # Calculate 'wins' from game data
        required_game_cols = {'game_id', 'home_team', 'away_team', 'home_score', 'away_score'}
        missing_game_cols = required_game_cols - set(game_df.columns)
        if missing_game_cols:
            raise KeyError(f"Game data is missing required columns: {missing_game_cols}")
        
        game_data_processed = game_df.copy()
        game_data_processed['winner'] = np.where(
            game_data_processed['home_score'] > game_data_processed['away_score'],
            game_data_processed['home_team'],
            np.where(
                game_data_processed['home_score'] < game_data_processed['away_score'],
                game_data_processed['away_team'],
                'Tie'  # Handle ties explicitly
            )
        )
        
        # Count wins per team, excluding ties
        wins = game_data_processed[game_data_processed['winner'] != 'Tie']['winner'].value_counts().reset_index()
        wins.columns = ['team', 'wins']
        
        # Merge wins into team_data
        team_data = team_data.merge(wins, on='team', how='left')
        team_data['wins'] = team_data['wins'].fillna(0).astype(int)
        
        # Optionally, handle ties as half-wins or other strategies
        if handle_ties in ['half', 'full']:
            ties = game_data_processed[game_data_processed['winner'] == 'Tie']
            if not ties.empty:
                tie_counts = pd.Series(ties['home_team'].tolist() + ties['away_team'].tolist()).value_counts().reset_index()
                tie_counts.columns = ['team', 'ties']
                team_data = team_data.merge(tie_counts, on='team', how='left')
                team_data['ties'] = team_data['ties'].fillna(0).astype(int)
                if handle_ties == 'half':
                    team_data['wins'] += 0.5 * team_data['ties']
                elif handle_ties == 'full':
                    team_data['wins'] += team_data['ties']
                logging.info(f"Ties have been handled using the '{handle_ties}' strategy.")
                print(f"Ties have been handled using the '{handle_ties}' strategy.")
            else:
                logging.info("No tied games found.")
                print("No tied games found.")
        else:
            logging.info("Ties have been ignored in win calculations.")
            print("Ties have been ignored in win calculations.")
        
        # Data Validation
        required_columns = [
            'team', 'fantasy_points', 'completions', 'passing_yards',
            'rushing_yards', 'receiving_yards', 'games', 'points',
            'avg_fantasy_points', 'avg_passing_yards',
            'avg_rushing_yards', 'avg_receiving_yards', 'wins'
        ]
        missing_columns = set(required_columns) - set(team_data.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns after cleaning: {missing_columns}")
        
        # Ensure no missing values in required columns
        if team_data[required_columns].isnull().any().any():
            raise ValueError("Missing values detected in required columns after cleaning.")
        
        # Check for multiple classes in 'wins'
        unique_wins = team_data['wins'].unique()
        if len(unique_wins) < 2:
            raise ValueError("Target variable 'wins' must contain at least two classes for classification.")
        
        logging.info("Data cleaning and aggregation completed successfully.")
        logging.info("Cleaned team-level data preview:")
        logging.info(f"\n{team_data.head()}")
        print("Data cleaning and aggregation completed successfully.")
        print("Cleaned team-level data preview:")
        print(team_data.head())
        return team_data
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        print(f"Error during data cleaning: {e}")
        return None

# Step 3: Model Training and Evaluation

def train_winner_prediction_model(df):
    """
    Trains a RandomForestClassifier to predict team performance based on wins.
    Implements hyperparameter tuning and cross-validation.
    """
    logging.info("Starting model training and evaluation...")
    try:
        # Define features
        feature_columns = ['points', 'avg_fantasy_points', 'avg_passing_yards',
                           'avg_rushing_yards', 'avg_receiving_yards']
        X = df[feature_columns]
        
        # Redefine target variable based on median wins
        median_wins = df['wins'].median()
        y = (df['wins'] > median_wins).astype(int)  # 1 if wins > median, else 0
        
        # Check the distribution of the target variable
        logging.info("Target variable distribution:")
        logging.info(f"\n{y.value_counts()}")
        print("Target variable distribution:")
        print(y.value_counts())
        
        # Handle cases where only one class is present (additional safety)
        if y.nunique() < 2:
            raise ValueError("The target variable 'y' must contain at least two classes for classification.")
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check if both classes are present in training and testing sets
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError("One of the splits has less than two classes. Consider adjusting the train-test split or handling class imbalance.")
        
        # Hyperparameter Tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        rf = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            scoring='roc_auc'
        )
        
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        logging.info(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best parameters found: {grid_search.best_params_}")
        
        # Predictions
        y_pred = best_rf.predict(X_test)
        y_proba = best_rf.predict_proba(X_test)[:, 1]
        
        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"ROC-AUC: {roc_auc:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        logging.info("\nModel Evaluation Metrics:")
        logging.info(f"Accuracy: {accuracy:.2f}")
        logging.info(f"Precision: {precision:.2f}")
        logging.info(f"Recall: {recall:.2f}")
        logging.info(f"F1-Score: {f1:.2f}")
        logging.info(f"ROC-AUC: {roc_auc:.2f}")
        logging.info("\nClassification Report:")
        logging.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
        
        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show(),1
        
        # Feature Importance
        feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis', hue=None, legend=False)
        plt.title('Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        logging.info("Feature importances plotted.")
        
        logging.info("Model training and evaluation completed successfully.")
        return best_rf
    except Exception as e:
        logging.error(f"Error during model training and evaluation: {e}")
        print(f"Error during model training and evaluation: {e}")
        return None

# Step 4: Betting Strategy

def predict_and_bet(df, model, threshold=0.75):
    """
    Predicts win probabilities and calculates bet amounts based on a probability threshold.

    Parameters:
    - df: DataFrame containing team data.
    - model: Trained model for predicting win probabilities.
    - threshold: Minimum win probability to place a bet.

    Returns:
    - List of betting results (team, win_probability, bet_amount).
    """
    logging.info("Starting prediction and betting strategy...")
    try:
        required_columns = ['points', 'avg_fantasy_points', 'avg_passing_yards',
                            'avg_rushing_yards', 'avg_receiving_yards']
        if not set(required_columns).issubset(df.columns):
            raise ValueError(f"DataFrame must contain the following columns for prediction: {required_columns}")

        X = df[required_columns]
        # Predict win probabilities
        df['win_probability'] = model.predict_proba(X)[:, 1]
        df_filtered = df[df['win_probability'] >= threshold]

        logging.info(f"Teams meeting the threshold ({threshold}): {len(df_filtered)}")
        print(f"Teams meeting the threshold ({threshold}): {len(df_filtered)}")

        betting_results = []
        for _, row in df_filtered.iterrows():
            team = row['team']
            win_prob = row['win_probability']

            # Dynamic bet sizing (e.g., percentage of bankroll proportional to confidence)
            bet_amount = win_prob * 100  # Example: Bet $100 scaled by win probability
            print(f"Place ${bet_amount:.2f} bet on {team} (Win Probability: {win_prob:.2f}).")

            betting_results.append((team, win_prob, bet_amount))
            logging.info(f"Team: {team}, Win Probability: {win_prob:.2f}, Bet Amount: ${bet_amount:.2f}")

        return betting_results
    except Exception as e:
        logging.error(f"Error during prediction and betting: {e}")
        print(f"Error during prediction and betting: {e}")
        return None

# Step 5: Visualization

def visualize_win_probabilities(df):
    """
    Visualizes the win probabilities of teams using a bar chart.
    """
    logging.info("Starting visualization of win probabilities...")
    try:
        if 'team' not in df.columns or 'win_probability' not in df.columns:
            raise ValueError("DataFrame must contain 'team' and 'win_probability' columns for visualization.")
        plt.figure(figsize=(14, 7))
        sns.barplot(x='team', y='win_probability', data=df, palette='viridis')
        plt.xlabel('Team')
        plt.ylabel('Win Probability')
        plt.title('Win Probabilities for Upcoming Week')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        logging.info("Visualization completed successfully.")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        print(f"Error during visualization: {e}")

# Step 6: Simulation

def simulate_betting(df, model, weeks=5, threshold=0.75):
    """
    Simulates betting over a specified number of weeks and calculates total profit.

    Parameters:
    - df: DataFrame containing team data.
    - model: Trained model for predicting win probabilities.
    - weeks: Number of weeks to simulate.
    - threshold: Minimum win probability to place a bet.

    Returns:
    - Total profit after simulation.
    """
    logging.info("Starting betting simulation...")
    try:
        total_profit = 0
        total_bets = 0
        for week in range(1, weeks + 1):
            logging.info(f"Simulating Week {week}...")
            print(f"\nSimulating Week {week}:")

            betting_results = predict_and_bet(df, model, threshold=threshold)
            if betting_results is None:
                raise ValueError("Failed to get betting results.")

            week_profit = 0
            for team, win_prob, bet_amount in betting_results:
                # Calculate profit: Assume $100 payout on win minus bet amount
                if win_prob > 0.5:  # Simulated outcome based on win probability
                    profit = bet_amount * (win_prob / (1 - win_prob)) - bet_amount
                else:
                    profit = -bet_amount

                week_profit += profit
                total_bets += 1

                logging.info(f"Week {week} - Team: {team}, Profit: ${profit:.2f}")

            print(f"Week {week} Profit: ${week_profit:.2f}")
            logging.info(f"Week {week} Profit: ${week_profit:.2f}")

            total_profit += week_profit

        roi = (total_profit / (total_bets * 100)) * 100 if total_bets > 0 else 0
        print(f"\nTotal Profit after {weeks} weeks: ${total_profit:.2f}")
        print(f"Total ROI: {roi:.2f}%")
        logging.info(f"Total Profit after {weeks} weeks: ${total_profit:.2f}")
        logging.info(f"Total ROI: {roi:.2f}%")
        return total_profit

    except Exception as e:
        logging.error(f"Error during simulation: {e}")
        print(f"Error during simulation: {e}")
        return None


# Main Execution Flow

def main():
    try:
        # Data Collection
        player_data = fetch_team_seasonal_data(years=[2023])
        game_data = fetch_pbp_data(years=[2023])  # Updated function
        roster_data = fetch_roster_data(years=[2023])
        
        # Data Cleaning and Aggregation
        team_data = clean_team_seasonal_data(player_data, game_data, roster_data, handle_ties='half')
        if team_data is None:
            print("Data cleaning failed. Exiting the program.")
            return
        
        # Model Training and Evaluation
        model = train_winner_prediction_model(team_data)
        if model is None:
            print("Model training failed. Exiting the program.")
            return
        
        # Betting Strategy
        betting_results = predict_and_bet(team_data, model)
        if betting_results is None:
            print("Betting strategy execution failed.")
        
        # Visualization
        if 'win_probability' in team_data.columns:
            visualize_win_probabilities(team_data)
        else:
            print("Win probabilities not available. Skipping visualization.")
        
        # Simulation
        simulate_betting(team_data, model, weeks=5)
        
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main execution flow: {e}")
        print(f"An unexpected error occurred: {e}")

# Execute the program
if __name__ == "__main__":
    main()
