#model-n.py
import os
import pandas as pd
import numpy as np
import textwrap
import google.generativeai as genai
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from google.api_core.exceptions import ResourceExhausted
from typing import Dict, List, Tuple, Optional

class PoliticalImpersonationDetector:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = 'gemini-2.0-flash',
        csv_file: str = "5525data.xlsx",
        output_dir: str = "results",
        checkpoint_file: str = "prediction_checkpoint.json",
        max_retries: int = 5,
        initial_retry_delay: int = 15
    ):
        """Initialize the impersonation detector with configuration.
        
        Args:
            api_key: Google API key (will use env var if None)
            model_name: Name of the Gemini model to use
            csv_file: Path to the input CSV file
            output_dir: Directory to save results
            checkpoint_file: File to save progress
            max_retries: Maximum number of retries for API calls
            initial_retry_delay: Initial delay between retries in seconds
        """
        # API Configuration
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY', 'AIzaSyDSTs3Ys4RKpejH2xLia8kwQN_NkkIlnyU')
        genai.configure(api_key=self.api_key)
        
        # Model Configuration
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # File Configuration
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.checkpoint_file = os.path.join(output_dir, checkpoint_file)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Retry Configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Prompt Templates (can be customized)
        self.prompt_templates = {
            "default": "請根據以下文本判斷其是否為反串。如果是，請回答 '1'；如果不是，請回答 '0'。\n文本如下：\n\n",
            "detailed": "請仔細分析以下政治評論，判斷這是真實支持者的言論還是反串。如果是反串（模仿支持者卻暗藏諷刺或試圖破壞），請回答'1'；如果是真實支持者的言論，請回答'0'。\n文本如下：\n\n",
            "few_shot": """請判斷以下政治評論是真實支持者還是反串。
例子1：「政府做得真好，我們的經濟已經完全崩潰了呢！」 評分：1（反串）
例子2：「對於這項政策我有疑慮，但我仍支持這個政黨的整體方向。」 評分：0（真實）
請針對下面的評論給出評分，1代表反串，0代表真實：
文本如下：\n\n"""
        }
        
        # Load Data
        self.df = pd.read_csv(csv_file, header=None)
        self.df.columns = ['content', 'label']
        
        # Initialize predictions list and start index
        self.predictions = []
        self.start_idx = 0
        
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists.
        
        Returns:
            bool: True if checkpoint was loaded, False otherwise
        """
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    self.predictions = checkpoint_data['predictions']
                    self.start_idx = checkpoint_data['last_processed_idx'] + 1
                    print(f"Resuming from index {self.start_idx}, found {len(self.predictions)} existing predictions")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                return False
        return False
    
    def save_checkpoint(self, idx: int) -> None:
        """Save checkpoint after processing an example.
        
        Args:
            idx: Index of the last processed example
        """
        checkpoint = {
            'predictions': self.predictions,
            'last_processed_idx': idx
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False)
    
    def predict_single(self, text: str, prompt_type: str = "default") -> str:
        """Generate a prediction for a single text example.
        
        Args:
            text: The text to classify
            prompt_type: The type of prompt to use from self.prompt_templates
            
        Returns:
            str: The prediction ('0' or '1')
        """
        prompt_template = self.prompt_templates.get(prompt_type, self.prompt_templates["default"])
        prompt = prompt_template + text
        
        retry_count = 0
        retry_delay = self.initial_retry_delay
        
        while retry_count < self.max_retries:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=5
                    )
                )
                
                answer = response.text.strip()
                # Ensure we only get '0' or '1'
                if answer not in ['0', '1']:
                    # Extract the first digit if there are multiple characters
                    for char in answer:
                        if char in ['0', '1']:
                            answer = char
                            break
                    else:
                        # If no 0 or 1 found, default to '0'
                        answer = '0'
                
                return answer
                
            except ResourceExhausted as e:
                retry_count += 1
                
                # Check if we received retry delay information in the error
                retry_delay_info = None
                if hasattr(e, 'details') and hasattr(e.details, 'retry_delay'):
                    retry_delay_info = e.details.retry_delay.seconds
                
                wait_time = retry_delay_info or retry_delay
                
                if retry_count < self.max_retries:
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{self.max_retries}...")
                    time.sleep(wait_time)
                    retry_delay *= 2
                else:
                    print("Maximum retries reached.")
                    raise
            
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                raise
        
        return '0'  # Default in case of persistent error
    
    def process_dataset(self, 
                         prompt_type: str = "default", 
                         test_size: float = 0.2, 
                         random_state: int = 42, 
                         use_existing_split: bool = False) -> pd.DataFrame:
        """Process the entire dataset with train/test split for evaluation.
        
        Args:
            prompt_type: The type of prompt to use
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            use_existing_split: Whether to use existing train/test split
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # First load any existing checkpoint
        self.load_checkpoint()
        
        # Create train/test split if not using existing
        if not use_existing_split:
            train_df, test_df = train_test_split(
                self.df, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=self.df['label']
            )
            test_df = test_df.reset_index(drop=True)
        else:
            test_df = self.df
        
        try:
            # Process each example in the test set
            for idx, row in test_df.iloc[self.start_idx:].iterrows():
                text = row['content']
                
                # Generate prediction
                answer = self.predict_single(text, prompt_type)
                self.predictions.append(answer)
                
                print(f"Prediction for text [{idx}]: {answer}\n{'-'*40}\n")
                
                # Save checkpoint after each prediction
                self.save_checkpoint(idx)
                
        except KeyboardInterrupt:
            print("Process interrupted by user. Progress has been saved.")
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            # Create results DataFrame
            result_df = test_df.copy()
            
            # Make sure predictions list is the right length
            while len(self.predictions) < len(result_df):
                self.predictions.append(None)
            
            # Limit predictions to length of dataframe (in case there are extra)
            self.predictions = self.predictions[:len(result_df)]
            
            # Add predictions to the DataFrame
            result_df['prediction'] = self.predictions
            
            return result_df
    
    def evaluate(self, result_df: pd.DataFrame) -> Dict:
        """Evaluate predictions against ground truth.
        
        Args:
            result_df: DataFrame with 'label' and 'prediction' columns
            
        Returns:
            Dict: Dictionary with evaluation metrics
        """
        # Filter out rows with None predictions
        eval_df = result_df.dropna(subset=['prediction'])
        
        # Convert predictions to integers
        y_true = eval_df['label'].astype(int)
        y_pred = eval_df['prediction'].astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate class-wise metrics
        class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Create an error analysis DataFrame
        error_df = eval_df[eval_df['label'] != eval_df['prediction'].astype(int)].copy()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_precision': class_metrics[0],
            'class_recall': class_metrics[1],
            'class_f1': class_metrics[2],
            'error_examples': error_df,
            'num_errors': len(error_df),
            'num_evaluated': len(eval_df)
        }
        
        return metrics
    
    def visualize_results(self, metrics: Dict, prompt_type: str = "default") -> None:
        """Visualize evaluation results.
        
        Args:
            metrics: Dictionary of evaluation metrics
            prompt_type: The prompt type used for these results
        """
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot confusion matrix
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Not Impersonation', 'Impersonation'],
                   yticklabels=['Not Impersonation', 'Impersonation'])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Confusion Matrix')
        
        # Plot metrics
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        values = [metrics[m] for m in metrics_to_plot]
        axes[1].bar(metrics_to_plot, values, color='skyblue')
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title(f'Performance Metrics (Prompt: {prompt_type})')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f'evaluation_{prompt_type}.png')
        plt.savefig(fig_path)
        print(f"Evaluation visualization saved to {fig_path}")
        
        # Display if in a notebook
        plt.show()
    
    def compare_prompts(self, prompt_types: List[str], test_size: float = 0.2) -> Dict:
        """Compare performance of different prompt types.
        
        Args:
            prompt_types: List of prompt types to compare
            test_size: Proportion of data to use for testing
            
        Returns:
            Dict: Dictionary with evaluation metrics for each prompt type
        """
        # Create train/test split
        train_df, test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=42, 
            stratify=self.df['label']
        )
        test_df = test_df.reset_index(drop=True)
        
        all_results = {}
        
        for prompt_type in prompt_types:
            print(f"\nEvaluating prompt: {prompt_type}")
            
            # Create a copy of the test set
            this_test_df = test_df.copy()
            
            # Reset predictions and start index for each prompt type
            self.predictions = []
            self.start_idx = 0
            
            # Process the test set with this prompt type
            checkpoint_file_original = self.checkpoint_file
            self.checkpoint_file = os.path.join(self.output_dir, f"checkpoint_{prompt_type}.json")
            
            result_df = self.process_dataset(
                prompt_type=prompt_type, 
                use_existing_split=True
            )
            
            # Evaluate results
            metrics = self.evaluate(result_df)
            
            # Visualize results
            self.visualize_results(metrics, prompt_type)
            
            # Store results
            all_results[prompt_type] = {
                'metrics': metrics,
                'result_df': result_df
            }
            
            # Save results
            result_df.to_csv(os.path.join(self.output_dir, f"results_{prompt_type}.csv"), index=False)
            
            # Restore original checkpoint file
            self.checkpoint_file = checkpoint_file_original
        
        # Create comparison chart
        self.visualize_prompt_comparison(all_results)
        
        return all_results
    
    def visualize_prompt_comparison(self, all_results: Dict) -> None:
        """Visualize comparison of prompt types.
        
        Args:
            all_results: Dictionary with results for each prompt type
        """
        # Extract metrics
        prompt_types = list(all_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create data for plot
        data = []
        for prompt_type in prompt_types:
            for metric in metrics:
                data.append({
                    'Prompt Type': prompt_type,
                    'Metric': metric,
                    'Value': all_results[prompt_type]['metrics'][metric]
                })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Prompt Type', y='Value', hue='Metric', data=comparison_df)
        plt.title('Comparison of Prompt Types')
        plt.ylim(0, 1.0)
        
        # Save figure
        comparison_path = os.path.join(self.output_dir, 'prompt_comparison.png')
        plt.savefig(comparison_path)
        print(f"Prompt comparison visualization saved to {comparison_path}")
        
        # Display if in a notebook
        plt.show()
    
    def cross_validation(self, n_splits: int = 5, prompt_type: str = "default") -> Dict:
        """Perform cross-validation evaluation.
        
        Args:
            n_splits: Number of cross-validation folds
            prompt_type: Prompt type to use
            
        Returns:
            Dict: Cross-validation results
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_metrics': [],
            'overall_metrics': {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }
        }
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.df)):
            print(f"\nProcessing fold {fold+1}/{n_splits}")
            
            # Get test set for this fold
            test_df = self.df.iloc[test_idx].reset_index(drop=True)
            
            # Reset predictions and start index
            self.predictions = []
            self.start_idx = 0
            
            # Update checkpoint file for this fold
            checkpoint_file_original = self.checkpoint_file
            self.checkpoint_file = os.path.join(self.output_dir, f"checkpoint_fold{fold+1}.json")
            
            # Process test set
            result_df = self.process_dataset(
                prompt_type=prompt_type,
                use_existing_split=True
            )
            
            # Evaluate results
            metrics = self.evaluate(result_df)
            
            # Store fold results
            cv_results['fold_metrics'].append(metrics)
            cv_results['overall_metrics']['accuracy'].append(metrics['accuracy'])
            cv_results['overall_metrics']['precision'].append(metrics['precision'])
            cv_results['overall_metrics']['recall'].append(metrics['recall'])
            cv_results['overall_metrics']['f1'].append(metrics['f1'])
            
            # Save results for this fold
            result_df.to_csv(os.path.join(self.output_dir, f"results_fold{fold+1}.csv"), index=False)
            
            # Restore original checkpoint file
            self.checkpoint_file = checkpoint_file_original
        
        # Calculate average metrics
        for metric in cv_results['overall_metrics']:
            values = cv_results['overall_metrics'][metric]
            cv_results['overall_metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # Visualize cross-validation results
        self.visualize_cv_results(cv_results, n_splits, prompt_type)
        
        return cv_results
    
    def visualize_cv_results(self, cv_results: Dict, n_splits: int, prompt_type: str) -> None:
        """Visualize cross-validation results.
        
        Args:
            cv_results: Cross-validation results
            n_splits: Number of cross-validation folds
            prompt_type: Prompt type used
        """
        # Create data for plot
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        means = [cv_results['overall_metrics'][m]['mean'] for m in metrics]
        stds = [cv_results['overall_metrics'][m]['std'] for m in metrics]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x, means, width, yerr=stds, alpha=0.7, capsize=10, label='Mean')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.0)
        plt.title(f'{n_splits}-Fold Cross-Validation Results (Prompt: {prompt_type})')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        
        # Add value labels
        for i, v in enumerate(means):
            plt.text(i, v + stds[i] + 0.02, f'{v:.3f}±{stds[i]:.3f}', ha='center')
        
        # Save figure
        cv_path = os.path.join(self.output_dir, f'cv_results_{prompt_type}.png')
        plt.savefig(cv_path)
        print(f"Cross-validation results visualization saved to {cv_path}")
        
        # Display if in a notebook
        plt.show()
    
    def error_analysis(self, result_df: pd.DataFrame, output_file: str = "error_analysis.csv") -> pd.DataFrame:
        """Perform error analysis on incorrect predictions.
        
        Args:
            result_df: DataFrame with predictions
            output_file: File to save error analysis
            
        Returns:
            pd.DataFrame: DataFrame with error analysis
        """
        # Filter to only include rows with predictions
        filtered_df = result_df.dropna(subset=['prediction'])
        
        # Convert prediction to same type as label
        filtered_df['prediction'] = filtered_df['prediction'].astype(int)
        
        # Find errors
        error_df = filtered_df[filtered_df['label'] != filtered_df['prediction']].copy()
        
        # Add error type column (false positive or false negative)
        error_df['error_type'] = error_df.apply(
            lambda row: 'False Positive' if row['prediction'] == 1 and row['label'] == 0 else 'False Negative', 
            axis=1
        )
        
        # Save error analysis
        error_path = os.path.join(self.output_dir, output_file)
        error_df.to_csv(error_path, index=False)
        print(f"Error analysis saved to {error_path}")
        
        # Print summary
        print("\nError Analysis Summary:")
        print(f"Total errors: {len(error_df)}")
        print(error_df['error_type'].value_counts())
        
        return error_df

# Example usage
if __name__ == "__main__":
    detector = PoliticalImpersonationDetector(
        csv_file="/Users/yui/Downloads/NLP_FP/NLP_FP.csv",
        output_dir="evaluation_results"
    )
    
    # Simple evaluation with default prompt
    result_df = detector.process_dataset(prompt_type="default", test_size=0.2)
    metrics = detector.evaluate(result_df)
    detector.visualize_results(metrics)
    
    # Compare different prompt templates
    comparison_results = detector.compare_prompts(
       prompt_types=["default", "detailed", "few_shot"]
    )
    
    # Perform cross-validation
    cv_results = detector.cross_validation(n_splits=5, prompt_type="default")
    
    # Perform error analysis
    error_df = detector.error_analysis(result_df)