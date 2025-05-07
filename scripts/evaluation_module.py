#!/usr/bin/env python3

import os
import json
import statistics
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

@dataclass
class NavigationTask:
    instruction: str
    start_position: Tuple[float, float, float]  # x, y, theta
    target_position: Tuple[float, float, float]  # x, y, theta
    completed: bool = False
    steps_taken: int = 0
    decision_times: List[float] = field(default_factory=list)
    success: bool = False
    end_position: Optional[Tuple[float, float, float]] = None

@dataclass
class EvaluationResult:
    gsr: float  # Goal Success Rate (percentage)
    avg_steps: float  # Average steps to completion
    avg_lpd: float  # Average Latency per Decision (ms)
    completed_tasks: int
    total_tasks: int
    task_results: List[Dict] = field(default_factory=list)

class NavigationEvaluator:
    def __init__(self, log_dir: str, success_threshold: float = 0.5):
        """
        Initialize the evaluator.
        
        Args:
            log_dir: Directory containing navigation logs
            success_threshold: Distance threshold (meters) to consider target reached
        """
        self.log_dir = log_dir
        self.success_threshold = success_threshold
        self.tasks = []
        
    def load_tasks_from_logs(self):
        """Load and parse navigation tasks from log files."""
        task_dirs = [d for d in os.listdir(self.log_dir) if os.path.isdir(os.path.join(self.log_dir, d))]
        
        for task_dir in task_dirs:
            task_path = os.path.join(self.log_dir, task_dir)
            task_config_path = os.path.join(task_path, "task_config.json")
            
            if not os.path.exists(task_config_path):
                print(f"Warning: No task config found in {task_path}")
                continue
                
            with open(task_config_path, 'r') as f:
                task_config = json.load(f)
                
            # Extract task information
            task = NavigationTask(
                instruction=task_config.get("instruction", ""),
                start_position=tuple(task_config.get("start_position", [0, 0, 0])),
                target_position=tuple(task_config.get("target_position", [0, 0, 0]))
            )
            
            # Process step directories to extract timing and decision data
            step_dirs = sorted([d for d in os.listdir(task_path) if d.startswith("step_")])
            
            for step_dir in step_dirs:
                step_path = os.path.join(task_path, step_dir)
                reply_path = os.path.join(step_path, "reply.txt")
                
                if os.path.exists(reply_path):
                    # Extract timestamps from filenames or metadata
                    step_timestamp = os.path.getmtime(step_path)
                    reply_timestamp = os.path.getmtime(reply_path)
                    decision_time = (reply_timestamp - step_timestamp) * 1000  # Convert to ms
                    task.decision_times.append(decision_time)
                    
                    # Check if task was completed in this step
                    with open(reply_path, 'r') as f:
                        reply_content = f.read()
                        if "----\nD - Stop" in reply_content:
                            task.completed = True
                            # If available, extract final position
                            position_file = os.path.join(step_path, "position.json")
                            if os.path.exists(position_file):
                                with open(position_file, 'r') as pos_f:
                                    pos_data = json.load(pos_f)
                                    task.end_position = (
                                        pos_data.get("x", 0),
                                        pos_data.get("y", 0),
                                        pos_data.get("theta", 0)
                                    )
            
            task.steps_taken = len(step_dirs)
            
            # Determine success based on completed flag and distance to target
            if task.completed and task.end_position:
                # Calculate distance to target
                target_dist = np.sqrt(
                    (task.end_position[0] - task.target_position[0])**2 +
                    (task.end_position[1] - task.target_position[1])**2
                )
                task.success = target_dist <= self.success_threshold
            
            self.tasks.append(task)
    
    def evaluate(self) -> EvaluationResult:
        """Evaluate all tasks and return metrics."""
        if not self.tasks:
            print("No tasks found for evaluation!")
            return EvaluationResult(gsr=0, avg_steps=0, avg_lpd=0, completed_tasks=0, total_tasks=0)
        
        # Count successful tasks
        successful_tasks = [task for task in self.tasks if task.success]
        completed_tasks = [task for task in self.tasks if task.completed]
        
        # Calculate metrics
        gsr = (len(successful_tasks) / len(self.tasks)) * 100
        
        avg_steps = 0
        if completed_tasks:
            avg_steps = sum(task.steps_taken for task in completed_tasks) / len(completed_tasks)
        
        # Calculate average latency per decision across all tasks
        all_decision_times = []
        for task in self.tasks:
            all_decision_times.extend(task.decision_times)
        
        avg_lpd = 0
        if all_decision_times:
            avg_lpd = sum(all_decision_times) / len(all_decision_times)
        
        # Create task results
        task_results = []
        for i, task in enumerate(self.tasks):
            task_results.append({
                "task_id": i,
                "instruction": task.instruction[:50] + "..." if len(task.instruction) > 50 else task.instruction,
                "completed": task.completed,
                "success": task.success,
                "steps": task.steps_taken,
                "avg_decision_time_ms": statistics.mean(task.decision_times) if task.decision_times else 0
            })
        
        return EvaluationResult(
            gsr=gsr,
            avg_steps=avg_steps,
            avg_lpd=avg_lpd,
            completed_tasks=len(completed_tasks),
            total_tasks=len(self.tasks),
            task_results=task_results
        )
    
    def generate_reports(self, result: EvaluationResult, output_dir: str = "evaluation_reports"):
        """Generate evaluation reports and visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        # Generate text report
        self._generate_text_report(result, output_dir)
    
    def _generate_text_report(self, result: EvaluationResult, output_dir: str):
        """Generate a text report with evaluation results."""
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=== ROBOT NAVIGATION EVALUATION REPORT ===\n\n")
            f.write(f"Total Tasks Evaluated: {result.total_tasks}\n")
            f.write(f"Tasks Completed: {result.completed_tasks}\n\n")
            
            f.write("=== KEY METRICS ===\n")
            f.write(f"Goal Success Rate (GSR): {result.gsr:.2f}%\n")
            f.write(f"Average Steps to Completion: {result.avg_steps:.2f}\n")
            f.write(f"Average Latency per Decision: {result.avg_lpd:.2f} ms\n\n")
            
            f.write("=== INDIVIDUAL TASK RESULTS ===\n")
            for task_result in result.task_results:
                f.write(f"Task {task_result['task_id'] + 1}:\n")
                f.write(f"  Instruction: {task_result['instruction']}\n")
                f.write(f"  Completed: {'Yes' if task_result['completed'] else 'No'}\n")
                f.write(f"  Success: {'Yes' if task_result['success'] else 'No'}\n")
                f.write(f"  Steps Taken: {task_result['steps']}\n")
                f.write(f"  Avg Decision Time: {task_result['avg_decision_time_ms']:.2f} ms\n\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate robot navigation performance')
    parser.add_argument('--log-dir', type=str, default='src/NLBot/scripts/tmp',
                        help='Directory containing navigation logs')
    parser.add_argument('--output-dir', type=str, default='evaluation_reports',
                        help='Directory to save evaluation reports')
    parser.add_argument('--success-threshold', type=float, default=0.5,
                        help='Distance threshold (meters) to consider target reached')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print(f"Starting evaluation of navigation logs in: {args.log_dir}")
    evaluator = NavigationEvaluator(args.log_dir, args.success_threshold)
    
    print("Loading task data from logs...")
    evaluator.load_tasks_from_logs()
    
    print("Evaluating navigation performance...")
    result = evaluator.evaluate()
    
    print("Generating evaluation reports...")
    evaluator.generate_reports(result, args.output_dir)
    
    print(f"Evaluation complete! Reports saved to: {args.output_dir}")
    print(f"Goal Success Rate (GSR): {result.gsr:.2f}%")
    print(f"Average Steps to Completion: {result.avg_steps:.2f}")
    print(f"Average Latency per Decision: {result.avg_lpd:.2f} ms")


if __name__ == "__main__":
    main()