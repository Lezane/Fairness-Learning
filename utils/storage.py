import os
import csv

class ExperimentRecorder:
    def __init__(self, output_dir, dataset_name):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, f"{dataset_name}_training_history.csv")
        
        # Write headers if the file is brand new
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'dataset', 'architecture', 'optimizer', 'run_id', 'seed', 
                    'epoch', 'train_s0', 'train_s1', 'test_s0', 'test_s1'
                ])

    def record_epoch(self, dataset, arch, opt_name, run_id, seed, epoch, tr_s0, tr_s1, te_s0, te_s1):
        """Appends a single epoch's metrics immediately to disk."""
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset, arch, opt_name, run_id, seed, 
                epoch, tr_s0, tr_s1, te_s0, te_s1
            ])