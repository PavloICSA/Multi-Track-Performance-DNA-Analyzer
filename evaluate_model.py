#!/usr/bin/env python3
"""
DNA Model Evaluation Script
Evaluate trained models on test datasets and compare model versions
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import json
import pandas as pd
import numpy as np
import torch

from dna_model_inference import DNAModelInference
from dna_feature_engineering import DNAFeatureEngineering
from dna_model_registry import DNAModelRegistry
from config import MODELS_DIR, LOGS_DIR, DNA_FEATURES, ARCHETYPE_CLASSES


def setup_logging(verbose: bool = False):
    """Configure logging for evaluation script"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    LOGS_DIR.mkdir(exist_ok=True)
    
    log_file = LOGS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained DNA prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate the latest model
  python evaluate_model.py
  
  # Evaluate a specific model version
  python evaluate_model.py --model-dir models/dna_model_v1.0.0
  
  # Evaluate on custom test data
  python evaluate_model.py --test-data ./test_tracks
  
  # Compare two model versions
  python evaluate_model.py --compare models/dna_model_v1.0.0 models/dna_model_v2.0.0
  
  # Generate detailed comparison report
  python evaluate_model.py --compare v1.0.0 v2.0.0 --report comparison_report.html
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Path to model directory (default: models/latest)'
    )
    
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('MODEL1', 'MODEL2'),
        help='Compare two model versions (provide version strings or paths)'
    )
    
    # Test data
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Directory containing test track data (default: use training data)'
    )
    
    parser.add_argument(
        '--test-csv',
        type=str,
        default=None,
        help='Single CSV file for testing'
    )
    
    # Output options
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Generate HTML evaluation report at specified path'
    )
    
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Save evaluation results as JSON'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    
    return parser.parse_args()


def list_available_models():
    """List all available trained models"""
    print("\n📦 Available Models:")
    print("=" * 70)
    
    registry = DNAModelRegistry()
    models = registry.list_models()
    
    if not models:
        print("No models found in registry.")
        return
    
    for model in models:
        version = model['version']
        metrics = model.get('metrics', {})
        is_prod = model.get('is_production', False)
        training_date = model.get('metadata', {}).get('training_date', 'Unknown')
        
        status = "🟢 PRODUCTION" if is_prod else "🔵 Available"
        
        print(f"\n{status} - Version {version}")
        print(f"   Trained: {training_date}")
        print(f"   Reliability: {metrics.get('overall_reliability', 0):.2%}")
        print(f"   Accuracy: {metrics.get('archetype_accuracy', 0):.2%}")
        print(f"   DNA R²: {metrics.get('dna_r2', 0):.3f}")


def load_test_data(test_data_path: str, test_csv: str = None):
    """Load test data from directory or CSV file"""
    feature_eng = DNAFeatureEngineering()
    
    if test_csv:
        # Load single CSV file
        df = feature_eng.load_and_process_csv(test_csv, track_name='test')
        return df
    
    if test_data_path:
        # Load from directory structure
        all_data = []
        test_path = Path(test_data_path)
        
        for csv_file in test_path.rglob('*.CSV'):
            if 'AnalysisEnduranceWithSections' in csv_file.name:
                track_name = csv_file.parent.parent.name if csv_file.parent.name.startswith('Race') else csv_file.parent.name
                df = feature_eng.load_and_process_csv(csv_file, track_name=track_name)
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
    
    return None


def evaluate_model(model_dir: str, test_data: pd.DataFrame, logger):
    """Evaluate a single model on test data"""
    logger.info(f"\n📊 Evaluating model: {model_dir}")
    
    try:
        # Load model
        inference = DNAModelInference(model_dir=model_dir)
        model_info = inference.get_model_info()
        
        logger.info(f"   Model version: {model_info['version']}")
        logger.info(f"   Training date: {model_info['training_date']}")
        
        # Process test data
        feature_eng = DNAFeatureEngineering()
        driver_features = feature_eng.extract_driver_features(test_data)
        
        logger.info(f"   Test samples: {len(driver_features)} driver-track combinations")
        
        # Generate predictions
        driver_profiles = inference.create_driver_profiles(test_data)
        
        logger.info(f"   ✅ Generated predictions for {len(driver_profiles)} drivers")
        
        # Calculate evaluation metrics if we have ground truth
        # For now, just return the predictions
        results = {
            'model_version': model_info['version'],
            'model_dir': model_dir,
            'training_date': model_info['training_date'],
            'training_metrics': model_info.get('validation_metrics', {}),
            'test_samples': len(driver_features),
            'predictions': len(driver_profiles),
            'driver_profiles': driver_profiles
        }
        
        return results
        
    except Exception as e:
        logger.error(f"   ❌ Evaluation failed: {e}")
        if logger.level == logging.DEBUG:
            import traceback
            logger.error(traceback.format_exc())
        return None


def compare_models(model1_path: str, model2_path: str, test_data: pd.DataFrame, logger):
    """Compare two model versions"""
    logger.info("\n🔍 Comparing Models")
    logger.info("=" * 70)
    
    # Evaluate both models
    results1 = evaluate_model(model1_path, test_data, logger)
    results2 = evaluate_model(model2_path, test_data, logger)
    
    if not results1 or not results2:
        logger.error("Failed to evaluate one or both models")
        return None
    
    # Compare metrics
    logger.info("\n📊 Comparison Results:")
    logger.info("=" * 70)
    
    metrics1 = results1['training_metrics']
    metrics2 = results2['training_metrics']
    
    comparison = {
        'model1': {
            'version': results1['model_version'],
            'path': model1_path,
            'metrics': metrics1
        },
        'model2': {
            'version': results2['model_version'],
            'path': model2_path,
            'metrics': metrics2
        },
        'differences': {}
    }
    
    # Compare key metrics
    metric_names = ['overall_reliability', 'archetype_accuracy', 'dna_r2', 'dna_mae']
    
    logger.info(f"\n{'Metric':<30} {'Model 1':<15} {'Model 2':<15} {'Difference':<15}")
    logger.info("-" * 75)
    
    for metric in metric_names:
        val1 = metrics1.get(metric, 0)
        val2 = metrics2.get(metric, 0)
        diff = val2 - val1
        
        # Format based on metric type
        if metric == 'dna_mae':
            # Lower is better for MAE
            better = "🟢" if diff < 0 else "🔴" if diff > 0 else "⚪"
            logger.info(f"{metric:<30} {val1:<15.4f} {val2:<15.4f} {diff:+.4f} {better}")
        else:
            # Higher is better for other metrics
            better = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
            logger.info(f"{metric:<30} {val1:<15.4f} {val2:<15.4f} {diff:+.4f} {better}")
        
        comparison['differences'][metric] = {
            'model1': val1,
            'model2': val2,
            'difference': diff
        }
    
    # Determine winner
    reliability_diff = comparison['differences']['overall_reliability']['difference']
    
    logger.info("\n" + "=" * 70)
    if reliability_diff > 0.01:
        logger.info(f"🏆 Model 2 ({results2['model_version']}) performs better")
        comparison['winner'] = 'model2'
    elif reliability_diff < -0.01:
        logger.info(f"🏆 Model 1 ({results1['model_version']}) performs better")
        comparison['winner'] = 'model1'
    else:
        logger.info("⚖️  Models perform similarly")
        comparison['winner'] = 'tie'
    
    return comparison


def generate_comparison_report(comparison: dict, output_path: str, logger):
    """Generate HTML comparison report"""
    logger.info(f"\n📄 Generating comparison report: {output_path}")
    
    model1 = comparison['model1']
    model2 = comparison['model2']
    diffs = comparison['differences']
    winner = comparison.get('winner', 'tie')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .model-card {{
            display: inline-block;
            width: 45%;
            margin: 10px;
            padding: 20px;
            border-radius: 8px;
            vertical-align: top;
        }}
        .model1 {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .model2 {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        .winner {{
            border: 4px solid gold;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .better {{
            color: #27ae60;
            font-weight: bold;
        }}
        .worse {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .neutral {{
            color: #95a5a6;
        }}
        .winner-badge {{
            display: inline-block;
            padding: 10px 20px;
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            color: white;
            border-radius: 25px;
            font-weight: bold;
            font-size: 18px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Model Comparison Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Models Under Comparison</h2>
        <div class="model-card model1 {'winner' if winner == 'model1' else ''}">
            <h3>Model 1</h3>
            <p><strong>Version:</strong> {model1['version']}</p>
            <p><strong>Path:</strong> {model1['path']}</p>
            {'<div class="winner-badge">🏆 WINNER</div>' if winner == 'model1' else ''}
        </div>
        
        <div class="model-card model2 {'winner' if winner == 'model2' else ''}">
            <h3>Model 2</h3>
            <p><strong>Version:</strong> {model2['version']}</p>
            <p><strong>Path:</strong> {model2['path']}</p>
            {'<div class="winner-badge">🏆 WINNER</div>' if winner == 'model2' else ''}
        </div>
        
        <h2>📊 Metric Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Model 1</th>
                <th>Model 2</th>
                <th>Difference</th>
                <th>Better</th>
            </tr>
"""
    
    for metric, values in diffs.items():
        val1 = values['model1']
        val2 = values['model2']
        diff = values['difference']
        
        # Determine which is better
        if metric == 'dna_mae':
            better_model = 'Model 2' if diff < 0 else 'Model 1' if diff > 0 else 'Tie'
            css_class = 'better' if diff < 0 else 'worse' if diff > 0 else 'neutral'
        else:
            better_model = 'Model 2' if diff > 0 else 'Model 1' if diff < 0 else 'Tie'
            css_class = 'better' if diff > 0 else 'worse' if diff < 0 else 'neutral'
        
        html_content += f"""
            <tr>
                <td>{metric}</td>
                <td>{val1:.4f}</td>
                <td>{val2:.4f}</td>
                <td class="{css_class}">{diff:+.4f}</td>
                <td>{better_model}</td>
            </tr>
"""
    
    html_content += f"""
        </table>
        
        <h2>🏆 Conclusion</h2>
        <p style="font-size: 18px;">
"""
    
    if winner == 'model1':
        html_content += f"Model 1 ({model1['version']}) performs better overall."
    elif winner == 'model2':
        html_content += f"Model 2 ({model2['version']}) performs better overall."
    else:
        html_content += "Both models perform similarly."
    
    html_content += """
        </p>
        
        <div style="margin-top: 40px; text-align: center; color: #7f8c8d;">
            Report generated by DNA Model Evaluation Script
        </div>
    </div>
</body>
</html>
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"   ✅ Report saved: {output_path}")


def main():
    """Main evaluation pipeline"""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    logger.info("=" * 70)
    logger.info("🧬 DNA Model Evaluation")
    logger.info("=" * 70)
    
    # List models if requested
    if args.list_models:
        list_available_models()
        return 0
    
    try:
        # Load test data
        test_data = None
        if args.test_data or args.test_csv:
            logger.info("\n📂 Loading test data...")
            test_data = load_test_data(args.test_data, args.test_csv)
            if test_data is None:
                logger.error("Failed to load test data")
                return 1
            logger.info(f"   ✅ Loaded {len(test_data)} test rows")
        else:
            logger.info("\n⚠️  No test data provided, using training data for evaluation")
            # Load from default location
            test_data = load_test_data('.', None)
            if test_data is None:
                logger.error("No data available for evaluation")
                return 1
        
        # Compare models if requested
        if args.compare:
            model1_path = args.compare[0]
            model2_path = args.compare[1]
            
            # Convert version strings to paths if needed
            if not Path(model1_path).exists():
                model1_path = str(MODELS_DIR / f"dna_model_{model1_path}")
            if not Path(model2_path).exists():
                model2_path = str(MODELS_DIR / f"dna_model_{model2_path}")
            
            comparison = compare_models(model1_path, model2_path, test_data, logger)
            
            if comparison and args.report:
                generate_comparison_report(comparison, args.report, logger)
            
            if comparison and args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(comparison, f, indent=2, default=str)
                logger.info(f"\n💾 Comparison results saved to: {args.output_json}")
        
        # Evaluate single model
        else:
            model_dir = args.model_dir or str(MODELS_DIR / "latest")
            results = evaluate_model(model_dir, test_data, logger)
            
            if results:
                logger.info("\n✅ Evaluation complete!")
                logger.info(f"   Model: {results['model_version']}")
                logger.info(f"   Test samples: {results['test_samples']}")
                logger.info(f"   Predictions: {results['predictions']}")
                
                if args.output_json:
                    # Remove driver_profiles from JSON output (too large)
                    output_results = {k: v for k, v in results.items() if k != 'driver_profiles'}
                    with open(args.output_json, 'w') as f:
                        json.dump(output_results, f, indent=2, default=str)
                    logger.info(f"\n💾 Results saved to: {args.output_json}")
            else:
                logger.error("Evaluation failed")
                return 1
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Evaluation completed successfully!")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
