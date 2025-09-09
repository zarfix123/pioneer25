#!/usr/bin/env python3
"""
Interactive Piece Influence Visualizer for East Asian Musical Influence Detection

Visualizes model predictions across segments of individual musical pieces.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load trained model and dataset."""
    model = joblib.load('/home/dennis/Projects/research/code/FINAL_MODEL.joblib')
    
    with open('/home/dennis/Projects/research/code/FINAL_MODEL_INFO.json', 'r') as f:
        model_info = json.load(f)
    
    western_data = pd.read_csv('/home/dennis/Projects/research/data/western data.csv')
    influenced_data = pd.read_csv('/home/dennis/Projects/research/data/influenced data.csv')
    combined_data = pd.concat([western_data, influenced_data], ignore_index=True)
    
    # Clean labels
    labels = combined_data['influence'].astype(str).str.extract(r'^\s*([01])')[0]
    combined_data['influence'] = pd.to_numeric(labels, errors='coerce')
    
    return model, model_info, combined_data

def get_available_pieces(data):
    """Get list of available pieces with segment counts."""
    pieces = []
    for piece in data['piece'].unique():
        count = len(data[data['piece'] == piece])
        pieces.append((piece, count))
    return pieces

def select_piece(pieces):
    """Let user select a piece to analyze."""
    print("\nAvailable pieces:")
    for i, (piece, count) in enumerate(pieces, 1):
        print(f"  {i:2}. {piece} ({count} segments)")
    
    while True:
        try:
            choice = input(f"\nSelect piece (1-{len(pieces)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(pieces):
                return pieces[idx][0]
            else:
                print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please enter a number.")

def analyze_piece(model, model_info, data, piece_name):
    """Analyze a specific piece."""
    piece_data = data[data['piece'] == piece_name].sort_values('start').reset_index(drop=True)
    
    if len(piece_data) == 0:
        print(f"No data found for {piece_name}")
        return
    
    # Make predictions
    feature_names = model_info['feature_names']
    X = piece_data[feature_names].values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    piece_data = piece_data.copy()
    piece_data['predicted_influence'] = predictions
    piece_data['confidence_non_influenced'] = probabilities[:, 0]
    piece_data['confidence_influenced'] = probabilities[:, 1]
    piece_data['max_confidence'] = np.max(probabilities, axis=1)
    
    return piece_data, feature_names

def create_influence_timeline(piece_data, piece_name):
    """Create influence prediction timeline."""
    plt.figure(1, figsize=(14, 6))
    plt.clf()
    
    segments = len(piece_data)
    times = range(segments)
    
    # Color segments based on predictions and confidence
    colors = []
    for _, row in piece_data.iterrows():
        if row['predicted_influence'] == 1:
            intensity = row['confidence_influenced']
            colors.append(plt.cm.Reds(0.3 + 0.7 * intensity))
        else:
            intensity = row['confidence_non_influenced']
            colors.append(plt.cm.Blues(0.3 + 0.7 * intensity))
    
    # Create bars
    bars = plt.bar(times, [1] * segments, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    
    # Add labels
    measure_labels = [f"{int(row['start'])}-{int(row['end'])}" for _, row in piece_data.iterrows()]
    plt.xticks(times, measure_labels, rotation=45, ha='right', fontsize=9)
    
    # Add confidence percentages and accuracy markers
    for i, (_, row) in enumerate(piece_data.iterrows()):
        confidence = row['max_confidence']
        plt.text(i, 0.5, f'{confidence:.0%}', ha='center', va='center', 
                fontweight='bold', fontsize=9, color='white')
        
        # Accuracy marker
        if row['influence'] == row['predicted_influence']:
            marker = '✓'
            color = 'green'
        else:
            marker = '✗'
            color = 'red'
        plt.text(i, 1.1, marker, ha='center', va='center', 
                fontsize=14, color=color, fontweight='bold')
    
    plt.ylabel('Influence Prediction')
    plt.title(f'East Asian Influence Timeline: {piece_name}', fontweight='bold', pad=20)
    plt.ylim(0, 1.3)
    plt.yticks([])
    
    # Legend
    red_patch = Rectangle((0, 0), 1, 1, facecolor=plt.cm.Reds(0.7), label='Influenced')
    blue_patch = Rectangle((0, 0), 1, 1, facecolor=plt.cm.Blues(0.7), label='Non-influenced')
    plt.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'/home/dennis/Projects/research/code/{piece_name}_timeline.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def create_confidence_plot(piece_data, piece_name):
    """Create confidence scores plot."""
    plt.figure(2, figsize=(14, 5))
    plt.clf()
    
    times = range(len(piece_data))
    confidence_influenced = piece_data['confidence_influenced'].values
    confidence_non_influenced = piece_data['confidence_non_influenced'].values
    
    plt.plot(times, confidence_influenced, 'ro-', label='Confidence: Influenced', 
            linewidth=2, markersize=6)
    plt.plot(times, confidence_non_influenced, 'bo-', label='Confidence: Non-influenced', 
            linewidth=2, markersize=6)
    
    plt.fill_between(times, confidence_influenced, alpha=0.3, color='red')
    plt.fill_between(times, confidence_non_influenced, alpha=0.3, color='blue')
    
    measure_labels = [f"{int(row['start'])}-{int(row['end'])}" for _, row in piece_data.iterrows()]
    plt.xticks(times, measure_labels, rotation=45, ha='right', fontsize=9)
    
    plt.ylabel('Confidence Score')
    plt.xlabel('Measures')
    plt.title(f'Model Confidence Scores: {piece_name}', fontweight='bold')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/home/dennis/Projects/research/code/{piece_name}_confidence.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_heatmap(piece_data, feature_names, piece_name):
    """Create feature values heatmap."""
    plt.figure(3, figsize=(14, 8))
    plt.clf()
    
    feature_data = piece_data[feature_names].values.T
    
    im = plt.imshow(feature_data, cmap='viridis', aspect='auto', alpha=0.8)
    
    plt.yticks(range(len(feature_names)), feature_names, fontsize=10)
    
    times = range(len(piece_data))
    measure_labels = [f"{int(row['start'])}-{int(row['end'])}" for _, row in piece_data.iterrows()]
    plt.xticks(times, measure_labels, rotation=45, ha='right', fontsize=9)
    
    plt.xlabel('Measures')
    plt.ylabel('Musical Features')
    plt.title(f'Feature Values Across Segments: {piece_name}', fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Feature Value (0-2)')
    
    plt.tight_layout()
    plt.savefig(f'/home/dennis/Projects/research/code/{piece_name}_features.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def print_analysis_summary(piece_data, piece_name):
    """Print detailed analysis summary."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY: {piece_name}")
    print(f"{'='*60}")
    
    total_segments = len(piece_data)
    influenced_segments = len(piece_data[piece_data['predicted_influence'] == 1])
    correct_predictions = len(piece_data[piece_data['influence'] == piece_data['predicted_influence']])
    
    print(f"Total segments: {total_segments}")
    print(f"Predicted influenced: {influenced_segments} ({influenced_segments/total_segments:.1%})")
    print(f"Prediction accuracy: {correct_predictions/total_segments:.1%}")
    print()
    
    print("Segment-by-segment results:")
    print("Measures | Actual      | Predicted   | Confidence | Status")
    print("-" * 55)
    
    for _, row in piece_data.iterrows():
        actual = "Influenced  " if row['influence'] == 1 else "Non-influenced"
        predicted = "Influenced  " if row['predicted_influence'] == 1 else "Non-influenced"
        confidence = row['max_confidence']
        status = "✓" if row['influence'] == row['predicted_influence'] else "✗"
        
        print(f"{int(row['start']):2}-{int(row['end']):2}     | {actual} | {predicted} | "
              f"{confidence:8.1%} | {status}")

def main():
    print("=== East Asian Influence Piece Visualizer ===")
    print("\nAnalyzes influence predictions for individual musical pieces")
    print("Creates three separate visualizations:")
    print("1. Influence timeline with confidence levels")
    print("2. Confidence scores over time")
    print("3. Feature values heatmap")
    
    try:
        # Load model and data
        print("\nLoading model and data...")
        model, model_info, data = load_model_and_data()
        
        # Get available pieces
        pieces = get_available_pieces(data)
        
        while True:
            # Let user select piece
            selected_piece = select_piece(pieces)
            
            print(f"\nAnalyzing {selected_piece}...")
            
            # Analyze piece
            piece_data, feature_names = analyze_piece(model, model_info, data, selected_piece)
            
            # Create visualizations
            create_influence_timeline(piece_data, selected_piece)
            create_confidence_plot(piece_data, selected_piece)
            create_feature_heatmap(piece_data, feature_names, selected_piece)
            
            # Print summary
            print_analysis_summary(piece_data, selected_piece)
            
            # Ask if user wants to analyze another piece
            while True:
                continue_analysis = input("\nAnalyze another piece? (y/n): ").strip().lower()
                if continue_analysis in ['y', 'yes']:
                    break
                elif continue_analysis in ['n', 'no']:
                    print("Analysis complete!")
                    return
                else:
                    print("Please enter 'y' or 'n'")
    
    except FileNotFoundError:
        print("Error: Model or data files not found. Please run train_model.py first.")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()