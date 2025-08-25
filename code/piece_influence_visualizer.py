import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class PieceInfluenceVisualizer:
    def __init__(self):
        """Initialize the visualizer with the trained model."""
        print("Loading trained model...")
        try:
            self.model = joblib.load('/home/dennis/Projects/research/code/FINAL_MODEL.joblib')
            self.model_info = joblib.load('/home/dennis/Projects/research/code/FINAL_MODEL_INFO.joblib')
            self.feature_names = self.model_info['feature_names']
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_data(self):
        """Load the combined dataset."""
        western_data = pd.read_csv('/home/dennis/Projects/research/data/western data.csv')
        influenced_data = pd.read_csv('/home/dennis/Projects/research/data/influenced data.csv')
        
        # Clean influenced data
        influenced_data['influence'] = pd.to_numeric(influenced_data['influence'], errors='coerce')
        influenced_data = influenced_data.dropna(subset=['influence'])
        
        combined_data = pd.concat([western_data, influenced_data], ignore_index=True)
        return combined_data
    
    def get_piece_data(self, piece_name):
        """Extract all segments for a specific piece."""
        data = self.load_data()
        piece_data = data[data['piece'].str.contains(piece_name, case=False, na=False)]
        
        if len(piece_data) == 0:
            print(f"No data found for piece: {piece_name}")
            print("Available pieces:")
            for piece in data['piece'].unique():
                print(f"  - {piece}")
            return None
        
        # Sort by start time
        piece_data = piece_data.sort_values('start').reset_index(drop=True)
        return piece_data
    
    def predict_piece_influence(self, piece_data):
        """Predict influence for all segments of a piece."""
        # Prepare features
        X = piece_data[self.feature_names].values
        
        # Get predictions and confidence
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Add predictions to dataframe
        results = piece_data.copy()
        results['predicted_influence'] = predictions
        results['confidence_non_influenced'] = probabilities[:, 0]
        results['confidence_influenced'] = probabilities[:, 1]
        results['max_confidence'] = np.max(probabilities, axis=1)
        
        return results
    
    def create_influence_heatmap(self, piece_results, piece_name):
        """Create a heatmap showing influence over time."""
        plt.figure(figsize=(16, 10))
        
        # Create main plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                           gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. MAIN INFLUENCE TIMELINE
        segments = len(piece_results)
        times = range(segments)
        
        # Create color map based on influence probability
        colors = []
        for _, row in piece_results.iterrows():
            if row['predicted_influence'] == 1:
                # Influenced - shades of red/orange
                intensity = row['confidence_influenced']
                colors.append(plt.cm.Reds(0.3 + 0.7 * intensity))
            else:
                # Non-influenced - shades of blue
                intensity = row['confidence_non_influenced']
                colors.append(plt.cm.Blues(0.3 + 0.7 * intensity))
        
        # Plot bars
        bars = ax1.bar(times, [1] * segments, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add measure labels
        measure_labels = [f"{row['start']}-{row['end']}" for _, row in piece_results.iterrows()]
        ax1.set_xticks(times)
        ax1.set_xticklabels(measure_labels, rotation=45, ha='right', fontsize=8)
        
        # Styling
        ax1.set_ylabel('Influence Prediction', fontsize=12, fontweight='bold')
        ax1.set_title(f'East Asian Influence Timeline: {piece_name}', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylim(0, 1.2)
        ax1.set_yticks([])
        
        # Add legend
        red_patch = Rectangle((0, 0), 1, 1, facecolor=plt.cm.Reds(0.7), label='Influenced')
        blue_patch = Rectangle((0, 0), 1, 1, facecolor=plt.cm.Blues(0.7), label='Non-influenced')
        ax1.legend(handles=[red_patch, blue_patch], loc='upper right')
        
        # Add actual vs predicted annotations
        for i, (_, row) in enumerate(piece_results.iterrows()):
            actual = row['influence']
            predicted = row['predicted_influence']
            confidence = row['max_confidence']
            
            # Add confidence text
            ax1.text(i, 0.5, f'{confidence:.0%}', ha='center', va='center', 
                    fontweight='bold', fontsize=9, color='white')
            
            # Add accuracy indicator
            if actual == predicted:
                marker = '✓'
                color = 'green'
            else:
                marker = '✗'
                color = 'red'
            ax1.text(i, 1.1, marker, ha='center', va='center', 
                    fontsize=12, color=color, fontweight='bold')
        
        # 2. CONFIDENCE SCORES
        confidence_influenced = piece_results['confidence_influenced'].values
        confidence_non_influenced = piece_results['confidence_non_influenced'].values
        
        ax2.plot(times, confidence_influenced, 'ro-', label='Confidence: Influenced', linewidth=2, markersize=4)
        ax2.plot(times, confidence_non_influenced, 'bo-', label='Confidence: Non-influenced', linewidth=2, markersize=4)
        ax2.fill_between(times, confidence_influenced, alpha=0.3, color='red')
        ax2.fill_between(times, confidence_non_influenced, alpha=0.3, color='blue')
        
        ax2.set_ylabel('Confidence', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(times)
        ax2.set_xticklabels(measure_labels, rotation=45, ha='right', fontsize=8)
        
        # 3. FEATURE HEATMAP
        feature_data = piece_results[self.feature_names].values.T
        im = ax3.imshow(feature_data, cmap='viridis', aspect='auto', alpha=0.8)
        
        ax3.set_yticks(range(len(self.feature_names)))
        ax3.set_yticklabels(self.feature_names, fontsize=10)
        ax3.set_xticks(times)
        ax3.set_xticklabels(measure_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_xlabel('Measures', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Features', fontsize=12, fontweight='bold')
        
        # Add colorbar for features
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Feature Value', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'/home/dennis/Projects/research/code/{piece_name}_influence_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def print_detailed_analysis(self, piece_results, piece_name):
        """Print detailed text analysis of the piece."""
        print(f"\n" + "="*70)
        print(f"DETAILED ANALYSIS: {piece_name}")
        print("="*70)
        
        total_segments = len(piece_results)
        influenced_segments = len(piece_results[piece_results['predicted_influence'] == 1])
        accuracy = len(piece_results[piece_results['influence'] == piece_results['predicted_influence']]) / total_segments
        
        print(f"Total segments: {total_segments}")
        print(f"Predicted influenced: {influenced_segments} ({influenced_segments/total_segments:.1%})")
        print(f"Prediction accuracy: {accuracy:.1%}")
        
        print(f"\n🎵 SEGMENT-BY-SEGMENT BREAKDOWN:")
        for i, (_, row) in enumerate(piece_results.iterrows()):
            actual = "Influenced" if row['influence'] == 1 else "Non-influenced"
            predicted = "Influenced" if row['predicted_influence'] == 1 else "Non-influenced"
            confidence = row['max_confidence']
            
            status = "✓" if row['influence'] == row['predicted_influence'] else "✗"
            
            print(f"  Measures {row['start']:2}-{row['end']:2} | "
                  f"Actual: {actual:13} | Predicted: {predicted:13} | "
                  f"Confidence: {confidence:.0%} | {status}")
        
        # Analyze prediction patterns
        print(f"\n🔍 INFLUENCE PATTERNS:")
        influenced_measures = piece_results[piece_results['predicted_influence'] == 1]
        if len(influenced_measures) > 0:
            print(f"Influenced sections found at measures:")
            for _, row in influenced_measures.iterrows():
                print(f"  - Measures {row['start']}-{row['end']} (confidence: {row['confidence_influenced']:.0%})")
        else:
            print("No influenced sections detected")
        
        # Feature analysis for influenced segments
        if len(influenced_measures) > 0:
            print(f"\n📊 KEY FEATURES IN INFLUENCED SECTIONS:")
            feature_avgs = influenced_measures[self.feature_names].mean()
            for feature, avg in feature_avgs.items():
                if avg > 0.5:  # Only show prominent features
                    print(f"  - {feature}: {avg:.1f} average")
    
    def analyze_piece(self, piece_name):
        """Complete analysis of a piece with visualization."""
        print(f"\n🎼 ANALYZING PIECE: {piece_name}")
        print("-" * 50)
        
        # Get piece data
        piece_data = self.get_piece_data(piece_name)
        if piece_data is None:
            return None
        
        print(f"Found {len(piece_data)} segments for {piece_name}")
        
        # Make predictions
        results = self.predict_piece_influence(piece_data)
        
        # Create visualizations
        self.create_influence_heatmap(results, piece_name)
        
        # Print detailed analysis
        self.print_detailed_analysis(results, piece_name)
        
        return results

def main():
    """Demo the visualizer with different pieces."""
    viz = PieceInfluenceVisualizer()
    
    print("🎵 PIECE INFLUENCE VISUALIZER")
    print("="*50)
    print("This tool analyzes how East Asian influence varies throughout a musical piece.")
    print("It shows predictions, confidence, and feature patterns over time.")
    
    # Get available pieces
    data = viz.load_data()
    pieces = data['piece'].unique()
    
    print(f"\nAvailable pieces ({len(pieces)}):")
    for i, piece in enumerate(pieces, 1):
        count = len(data[data['piece'] == piece])
        print(f"  {i:2}. {piece} ({count} segments)")
    
    print(f"\n" + "="*50)
    
    # Analyze a few example pieces
    example_pieces = ['Pagodes', 'Moonlight', 'Waldstein']
    
    for piece in example_pieces:
        if piece in [p for p in pieces if piece.lower() in p.lower()]:
            viz.analyze_piece(piece)
            print("\n" + "="*70 + "\n")
        else:
            print(f"Piece '{piece}' not found in dataset")

if __name__ == "__main__":
    main() 