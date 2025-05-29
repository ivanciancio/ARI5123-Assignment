"""
Benchmark Runner Script

This script runs the complete benchmark comparison against Sezer & Ozbayoglu (2018).
It can be run independently from the command line.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.feature_engineering import FeatureEngineer
from utils.model import CNNTradingModel
from utils.trading_strategy import TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Class to run comprehensive benchmark experiments against Sezer & Ozbayoglu (2018).
    """
    
    def __init__(self, output_dir="benchmark_results"):
        """
        Initialise the benchmark runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Benchmark paper results for comparison
        self.benchmark_paper_results = {
            "2007-2012": {
                'classification_accuracy': 0.552,
                'cnn_return': 0.072,
                'bah_return': 0.0586,
                'cnn_drawdown': 0.272,
                'bah_drawdown': 0.352,
                'success_rate': 0.767
            },
            "2012-2017": {
                'classification_accuracy': 0.538,
                'cnn_return': 0.0584,
                'bah_return': 0.1325,
                'cnn_drawdown': 0.195,
                'bah_drawdown': 0.0697,
                'success_rate': 0.448
            }
        }
    
    def create_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def run_single_stock_experiment(self, ticker, test_period, quick_run=False):
        """
        Run experiment for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            test_period: Either "2007-2012" or "2012-2017"
            quick_run: If True, use reduced epochs for faster testing
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Starting experiment for {ticker} ({test_period})")
        
        try:
            # Initialize components
            data_loader = DataLoader()
            feature_engineer = FeatureEngineer()
            
            # Download data
            logger.info(f"Downloading data for {ticker}")
            data = data_loader.download_benchmark_data(force_download=False)
            
            if ticker not in data:
                raise ValueError(f"No data available for {ticker}")
            
            # Get benchmark splits
            splits = data_loader.get_benchmark_splits(data, ticker, test_period)
            logger.info(f"Data split - Train: {len(splits['train_data'])}, Test: {len(splits['test_data'])}")
            
            # Prepare features
            logger.info("Preparing benchmark features")
            prepared_data = feature_engineer.prepare_benchmark_features(
                splits['train_data'], splits['test_data'], ticker
            )
            
            # Train model
            logger.info("Training CNN model")
            model = CNNTradingModel()
            model.build_simple_cnn()  # Use simple CNN as per benchmark
            
            # Training configuration
            epochs = 30 if quick_run else 100
            history = model.train(
                prepared_data['X_train'], prepared_data['y_train'],
                prepared_data['X_val'], prepared_data['y_val'],
                epochs=epochs,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=1e-5
            )
            
            # Evaluate model
            logger.info("Evaluating model performance")
            model_metrics = model.evaluate(prepared_data['X_test'], prepared_data['y_test'])
            y_pred_prob = model.predict(prepared_data['X_test'])
            
            # Run trading strategy
            logger.info("Running trading strategy")
            strategy = TradingStrategy(initial_capital=10000, transaction_cost=0.001)
            
            # Prepare trading data
            test_data = splits['test_data']
            test_prices = test_data['Close'].values
            test_dates = test_data.index
            
            # Align predictions with prices
            min_len = min(len(y_pred_prob), len(test_prices))
            signals = y_pred_prob[:min_len]
            prices = test_prices[:min_len]
            dates = test_dates[:min_len]
            
            # Execute strategies
            strategy_results = strategy.apply_strategy_optimized(
                prices, signals, dates,
                ticker=ticker,
                test_period=test_period
            )
            
            # Reset strategy for benchmark
            strategy.reset()
            benchmark_results = strategy.apply_buy_and_hold(prices)
            
            # Compile results
            experiment_results = {
                'ticker': ticker,
                'test_period': test_period,
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'train_samples': len(prepared_data['X_train']),
                    'val_samples': len(prepared_data['X_val']),
                    'test_samples': len(prepared_data['X_test']),
                    'train_period': splits['train_dates'],
                    'test_period_dates': splits['test_dates']
                },
                'model_performance': model_metrics,
                'trading_results': {
                    'cnn_strategy': strategy_results,
                    'buy_and_hold': benchmark_results
                },
                'training_history': {
                    'epochs_completed': len(history['loss']),
                    'final_train_loss': history['loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'final_train_acc': history['accuracy'][-1],
                    'final_val_acc': history['val_accuracy'][-1]
                }
            }
            
            logger.info(f"Completed experiment for {ticker}")
            logger.info(f"  Classification Accuracy: {model_metrics['accuracy']:.1%}")
            logger.info(f"  Strategy Return: {strategy_results['total_return']:.2%}")
            logger.info(f"  Buy & Hold Return: {benchmark_results['total_return']:.2%}")
            logger.info(f"  Outperformance: {strategy_results['total_return'] - benchmark_results['total_return']:.2%}")
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"Error in experiment for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'test_period': test_period,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_batch_experiment(self, test_period="2007-2012", stocks=None, max_stocks=20, quick_run=False):
        """
        Run batch experiment across multiple stocks.
        
        Args:
            test_period: Either "2007-2012" or "2012-2017"
            stocks: List of stock tickers (None for default selection)
            max_stocks: Maximum number of stocks to test
            quick_run: If True, use reduced epochs for faster testing
            
        Returns:
            Dictionary with batch results
        """
        # Enhanced default stock selection - full Dow 30 from benchmark period
        if stocks is None:
            stocks = [
                "AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS",
                "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
                "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"
            ][:max_stocks]
        
        logger.info(f"Starting comprehensive batch experiment for {test_period}")
        logger.info(f"Testing {len(stocks)} stocks: {', '.join(stocks)}")
        
        batch_results = {
            'test_period': test_period,
            'stocks_tested': len(stocks),
            'timestamp': datetime.now().isoformat(),
            'individual_results': {},
            'summary_statistics': {},
            'benchmark_comparison': {}
        }
        
        successful_results = []
        
        for i, ticker in enumerate(stocks):
            logger.info(f"Processing {ticker} ({i+1}/{len(stocks)})")
            
            result = self.run_single_stock_experiment(ticker, test_period, quick_run)
            batch_results['individual_results'][ticker] = result
            
            if 'error' not in result:
                successful_results.append(result)
        
        # Generate summary statistics
        if successful_results:
            batch_results['summary_statistics'] = self.calculate_batch_summary(successful_results)
            batch_results['benchmark_comparison'] = self.compare_with_benchmark_paper(
                batch_results['summary_statistics'], test_period
            )
        
        # Save results
        self.save_batch_results(batch_results, test_period)
        
        logger.info(f"Batch experiment completed. {len(successful_results)}/{len(stocks)} stocks successful")
        
        return batch_results
    
    def calculate_batch_summary(self, results):
        """
        Calculate summary statistics from batch results.
        
        Args:
            results: List of individual experiment results
            
        Returns:
            Dictionary with summary statistics
        """
        # Extract metrics
        accuracies = [r['model_performance']['accuracy'] for r in results]
        strategy_returns = [r['trading_results']['cnn_strategy']['total_return'] for r in results]
        benchmark_returns = [r['trading_results']['buy_and_hold']['total_return'] for r in results]
        strategy_drawdowns = [r['trading_results']['cnn_strategy']['max_drawdown'] for r in results]
        benchmark_drawdowns = [r['trading_results']['buy_and_hold']['max_drawdown'] for r in results]
        sharpe_ratios = [r['trading_results']['cnn_strategy']['sharpe_ratio'] for r in results]
        
        # Calculate outperformance
        outperformances = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
        success_count = sum(1 for x in outperformances if x > 0)
        success_rate = success_count / len(results)
        
        summary = {
            'num_successful_stocks': len(results),
            'classification_metrics': {
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'above_random_count': sum(1 for acc in accuracies if acc > 0.33)
            },
            'trading_performance': {
                'avg_strategy_return': np.mean(strategy_returns),
                'avg_benchmark_return': np.mean(benchmark_returns),
                'avg_outperformance': np.mean(outperformances),
                'success_rate': success_rate,
                'avg_strategy_drawdown': np.mean(strategy_drawdowns),
                'avg_benchmark_drawdown': np.mean(benchmark_drawdowns),
                'avg_sharpe_ratio': np.mean(sharpe_ratios)
            }
        }
        
        return summary
    
    def compare_with_benchmark_paper(self, summary_stats, test_period):
        """
        Compare batch results with benchmark paper.
        
        Args:
            summary_stats: Summary statistics from batch experiment
            test_period: Test period used
            
        Returns:
            Dictionary with benchmark paper comparison
        """
        paper_results = self.benchmark_paper_results[test_period]
        
        return {
            'accuracy_comparison': {
                'our_avg': summary_stats['classification_metrics']['avg_accuracy'],
                'paper_avg': paper_results['classification_accuracy'],
                'difference': summary_stats['classification_metrics']['avg_accuracy'] - paper_results['classification_accuracy']
            },
            'success_rate_comparison': {
                'our_rate': summary_stats['trading_performance']['success_rate'],
                'paper_rate': paper_results['success_rate'],
                'difference': summary_stats['trading_performance']['success_rate'] - paper_results['success_rate']
            },
            'return_comparison': {
                'our_strategy_avg': summary_stats['trading_performance']['avg_strategy_return'],
                'paper_strategy_avg': paper_results['cnn_return'],
                'our_benchmark_avg': summary_stats['trading_performance']['avg_benchmark_return'],
                'paper_benchmark_avg': paper_results['bah_return']
            }
        }
    
    def save_batch_results(self, batch_results, test_period):
        """
        Save batch experiment results to files.
        
        Args:
            batch_results: Batch experiment results
            test_period: Test period used
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results as JSON
        results_file = os.path.join(
            self.output_dir, 
            f"benchmark_results_{test_period}_{timestamp}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        logger.info(f"Saved complete results to: {results_file}")
        
        # Save summary as CSV
        if batch_results['individual_results']:
            summary_data = []
            for ticker, result in batch_results['individual_results'].items():
                if 'error' not in result:
                    summary_data.append({
                        'Ticker': ticker,
                        'Classification_Accuracy': result['model_performance']['accuracy'],
                        'Strategy_Return': result['trading_results']['cnn_strategy']['total_return'],
                        'BuyHold_Return': result['trading_results']['buy_and_hold']['total_return'],
                        'Outperformance': result['trading_results']['cnn_strategy']['total_return'] - 
                                        result['trading_results']['buy_and_hold']['total_return'],
                        'Strategy_Drawdown': result['trading_results']['cnn_strategy']['max_drawdown'],
                        'BuyHold_Drawdown': result['trading_results']['buy_and_hold']['max_drawdown'],
                        'Sharpe_Ratio': result['trading_results']['cnn_strategy']['sharpe_ratio'],
                        'Num_Trades': result['trading_results']['cnn_strategy']['number_of_trades']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_file = os.path.join(
                    self.output_dir,
                    f"benchmark_summary_{test_period}_{timestamp}.csv"
                )
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"Saved summary to: {summary_file}")

def main():
    """Main function for running benchmark experiments from command line."""
    parser = argparse.ArgumentParser(description='Run CNN-BI Benchmark Experiments')
    parser.add_argument('--period', choices=['2007-2012', '2012-2017', 'both'], 
                       default='2007-2012', help='Test period to run')
    parser.add_argument('--stocks', nargs='+', default=None,
                       help='Specific stocks to test (default: AAPL MSFT JPM JNJ KO)')
    parser.add_argument('--max-stocks', type=int, default=5,
                       help='Maximum number of stocks to test')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick experiment with reduced epochs')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(output_dir=args.output_dir)
    
    # Run experiments
    if args.period == 'both':
        periods = ['2007-2012', '2012-2017']
    else:
        periods = [args.period]
    
    for period in periods:
        logger.info(f"Running benchmark experiment for period: {period}")
        
        results = runner.run_batch_experiment(
            test_period=period,
            stocks=args.stocks,
            max_stocks=args.max_stocks,
            quick_run=args.quick
        )
        
        if results['summary_statistics']:
            print(f"\n=== Results Summary for {period} ===")
            print(f"Average Classification Accuracy: {results['summary_statistics']['classification_metrics']['avg_accuracy']:.1%}")
            print(f"Success Rate vs Buy & Hold: {results['summary_statistics']['trading_performance']['success_rate']:.1%}")
            print(f"Average Outperformance: {results['summary_statistics']['trading_performance']['avg_outperformance']:.2%}")
        
        print(f"Detailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()