"""
COMPLETE EXPERIMENT SUITE FOR ALL SAMPLERS
===========================================
Automated training, evaluation, and analysis for research paper

File: run_all_experiments.py
"""

import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import time


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Primary experiments (for main comparison table)
PRIMARY_EXPERIMENTS = {
    "baseline_pdf": {
        "sampler": "pdf",
        "description": "Nerfstudio default PDF sampling",
        "color": "gray",
    },
    "baseline_l0": {
        "sampler": "l0",
        "use_blur": True,
        "description": "L0-Sampler (Li et al. 2023)",
        "color": "blue",
    },
    "akm_v2_epanechnikov": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "epanechnikov",
        "use_blur": True,
        "description": "AKM-v2 Epanechnikov (OURS - MAIN)",
        "color": "red",
    },
    "akm_v2_gaussian": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "gaussian",
        "use_blur": True,
        "description": "AKM-v2 Gaussian",
        "color": "orange",
    },
}

# Secondary experiments (for extended comparison)
SECONDARY_EXPERIMENTS = {
    "akm_v3_epanechnikov": {
        "sampler": "adaptive_kernel_v3",
        "kernel_type": "epanechnikov",
        "description": "AKM-v3 with Entropy Adaptation",
    },
    "gmm_sampler": {
        "sampler": "gmm",
        "K": 4,
        "description": "Gaussian Mixture Model (K=4)",
    },
    "optimal_transport": {
        "sampler": "optimal_transport",
        "description": "Optimal Transport Sampler",
    },
    "entropy_kde": {
        "sampler": "entropy_kde",
        "kernel_type": "gaussian",
        "description": "Entropy-KDE Sampler",
    },
    "wavelet_hierarchical": {
        "sampler": "wavelet",
        "levels": 3,
        "description": "Wavelet Hierarchical (3 levels)",
    },
    "kernel_tilted": {
        "sampler": "kernel_tilted",
        "kernel": "gaussian",
        "description": "Kernel-Tilted (K-TOSS)",
    },
}

# Ablation studies
ABLATION_BLUR = {
    "akm_no_blur": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "epanechnikov",
        "use_blur": False,
        "description": "AKM without maxblur",
    },
    "akm_with_blur": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "epanechnikov",
        "use_blur": True,
        "description": "AKM with maxblur",
    },
}

ABLATION_KERNEL = {
    "akm_gaussian": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "gaussian",
        "use_blur": True,
    },
    "akm_epanechnikov": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "epanechnikov",
        "use_blur": True,
    },
    "akm_triangular": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "triangular",
        "use_blur": True,
    },
    "akm_uniform": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "uniform",
        "use_blur": True,
    },
}

ABLATION_SAMPLES = {
    "akm_32_samples": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "epanechnikov",
        "num_samples": 32,
    },
    "akm_64_samples": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "epanechnikov",
        "num_samples": 64,
    },
    "akm_128_samples": {
        "sampler": "adaptive_kernel_v2",
        "kernel_type": "epanechnikov",
        "num_samples": 128,
    },
}

# Dataset configurations
DATASETS = {
    "blender": {
        "scenes": ["lego", "chair", "drums", "ship", "mic", "ficus"],
        "path": "data/blender",
        "iterations": 30000,
        "eval_interval": 5000,
    },
    "llff": {
        "scenes": ["fern", "flower", "horns", "room"],
        "path": "data/nerf_llff_data",
        "iterations": 30000,
        "eval_interval": 5000,
    },
}

# Quick test config (for verification)
QUICK_TEST = {
    "dataset": "blender",
    "scenes": ["lego"],
    "iterations": 5000,
    "experiments": ["baseline_pdf", "baseline_l0", "akm_v2_epanechnikov"],
}


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def build_train_command(
    experiment_name: str,
    config: Dict,
    dataset: str,
    scene: str,
    iterations: int,
    output_dir: str = "outputs",
    data_dir: str = "data",
) -> List[str]:
    """Build nerfstudio training command"""
    
    dataset_path = f"{data_dir}/{dataset}/{scene}"
    
    cmd = [
        "ns-train", "nerfacto",
        "--data", dataset_path,
        "--output-dir", f"{output_dir}/{experiment_name}",
        "--experiment-name", scene,
        "--max-num-iterations", str(iterations),
        "--steps-per-save", "5000",
        "--steps-per-eval-image", str(iterations // 6),
        "--pipeline.datamanager.train-num-rays-per-batch", "4096",
        "--viewer.quit-on-train-completion", "True",
        "--logging.local-writer.max-log-size", "0",
    ]
    
    # Add sampler-specific configuration
    sampler_type = config.get("sampler", "pdf")
    
    if sampler_type != "pdf":
        cmd.extend(["--pipeline.model.proposal-sampler", sampler_type])
    
    # Kernel type
    if "kernel_type" in config:
        cmd.extend(["--pipeline.model.kernel-type", config["kernel_type"]])
    
    # Blur
    if "use_blur" in config:
        cmd.extend(["--pipeline.model.use-blur", str(config["use_blur"]).lower()])
    
    # Number of samples
    if "num_samples" in config:
        cmd.extend(["--pipeline.model.num-nerf-samples-per-ray", str(config["num_samples"])])
    
    # GMM K
    if "K" in config:
        cmd.extend(["--pipeline.model.gmm-components", str(config["K"])])
    
    # Wavelet levels
    if "levels" in config:
        cmd.extend(["--pipeline.model.wavelet-levels", str(config["levels"])])
    
    return cmd


def run_single_experiment(
    experiment_name: str,
    config: Dict,
    dataset: str,
    scene: str,
    iterations: int,
    verbose: bool = True,
) -> Dict:
    """Run single experiment and return results"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🚀 Running: {experiment_name} / {dataset} / {scene}")
        print(f"   {config.get('description', '')}")
        print(f"{'='*80}\n")
    
    cmd = build_train_command(experiment_name, config, dataset, scene, iterations)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=not verbose)
    elapsed_time = time.time() - start_time
    
    success = result.returncode == 0
    
    if verbose:
        if success:
            print(f"✅ Completed in {elapsed_time/60:.1f} minutes")
        else:
            print(f"❌ Failed after {elapsed_time/60:.1f} minutes")
    
    return {
        "experiment": experiment_name,
        "dataset": dataset,
        "scene": scene,
        "success": success,
        "time_minutes": elapsed_time / 60,
        "command": " ".join(cmd),
    }


def run_experiment_suite(
    experiments: Dict,
    datasets: Dict,
    output_file: str = "experiment_log.json",
    verbose: bool = True,
):
    """Run full suite of experiments"""
    
    results = []
    total = sum(len(ds["scenes"]) for ds in datasets.values()) * len(experiments)
    completed = 0
    
    print(f"\n🎯 Starting experiment suite: {total} total runs\n")
    
    for exp_name, exp_config in experiments.items():
        for dataset_name, dataset_config in datasets.items():
            for scene in dataset_config["scenes"]:
                
                result = run_single_experiment(
                    experiment_name=exp_name,
                    config=exp_config,
                    dataset=dataset_name,
                    scene=scene,
                    iterations=dataset_config["iterations"],
                    verbose=verbose,
                )
                
                results.append(result)
                completed += 1
                
                # Save incremental results
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\n📊 Progress: {completed}/{total} ({100*completed/total:.1f}%)\n")
    
    return results


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_experiment(
    experiment_name: str,
    dataset: str,
    scene: str,
    output_dir: str = "outputs",
) -> Dict:
    """Evaluate single experiment"""
    
    config_path = f"{output_dir}/{experiment_name}/{scene}/nerfacto/{scene}/config.yml"
    metrics_path = f"{output_dir}/{experiment_name}/{scene}/metrics.json"
    
    if not os.path.exists(config_path):
        print(f"⚠️  Config not found: {config_path}")
        return None
    
    cmd = [
        "ns-eval",
        "--load-config", config_path,
        "--output-path", metrics_path,
    ]
    
    print(f"📊 Evaluating: {experiment_name}/{scene}")
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        return {
            "experiment": experiment_name,
            "dataset": dataset,
            "scene": scene,
            **metrics
        }
    else:
        print(f"❌ Evaluation failed: {experiment_name}/{scene}")
        return None


def evaluate_all_experiments(
    experiments: Dict,
    datasets: Dict,
    output_file: str = "all_metrics.json",
):
    """Evaluate all experiments"""
    
    all_metrics = []
    
    for exp_name in experiments.keys():
        for dataset_name, dataset_config in datasets.items():
            for scene in dataset_config["scenes"]:
                
                metrics = evaluate_experiment(exp_name, dataset_name, scene)
                
                if metrics:
                    all_metrics.append(metrics)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_file.replace('.json', '.csv'), index=False)
    
    print(f"\n✅ Metrics saved to {output_file}")
    
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per experiment"""
    
    summary = df.groupby(['experiment', 'dataset']).agg({
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std'],
        'lpips': ['mean', 'std'],
    }).round(3)
    
    return summary


def generate_latex_table(df: pd.DataFrame, output_file: str = "results_table.tex"):
    """Generate LaTeX table for paper"""
    
    # Pivot for better presentation
    pivot = df.pivot_table(
        values=['psnr', 'ssim', 'lpips'],
        index=['dataset', 'scene'],
        columns='experiment',
        aggfunc='mean'
    )
    
    with open(output_file, 'w') as f:
        f.write("% Auto-generated results table\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Quantitative comparison of sampling methods}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(pivot.to_latex(float_format="%.2f"))
        f.write("}\n")
        f.write("\\end{table*}\n")
    
    print(f"📄 LaTeX table saved to {output_file}")


def compute_statistical_significance(df: pd.DataFrame, baseline: str = "baseline_pdf"):
    """Compute t-test for statistical significance"""
    
    from scipy import stats
    
    results = []
    
    baseline_data = df[df['experiment'] == baseline]
    
    for exp in df['experiment'].unique():
        if exp == baseline:
            continue
        
        exp_data = df[df['experiment'] == exp]
        
        # Merge on scene for paired t-test
        merged = baseline_data.merge(
            exp_data,
            on=['dataset', 'scene'],
            suffixes=('_baseline', '_exp')
        )
        
        if len(merged) > 0:
            t_stat, p_value = stats.ttest_rel(
                merged['psnr_exp'],
                merged['psnr_baseline']
            )
            
            results.append({
                'experiment': exp,
                'vs_baseline': baseline,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'improvement_db': (merged['psnr_exp'] - merged['psnr_baseline']).mean()
            })
    
    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(df: pd.DataFrame, output_dir: str = "plots"):
    """Generate all plots for paper"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 1. PSNR comparison
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='scene', y='psnr', hue='experiment')
    plt.title('PSNR Comparison Across Methods and Scenes', fontsize=14, fontweight='bold')
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.xlabel('Scene', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/psnr_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SSIM comparison
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='scene', y='ssim', hue='experiment')
    plt.title('SSIM Comparison Across Methods and Scenes', fontsize=14, fontweight='bold')
    plt.ylabel('SSIM', fontsize=12)
    plt.xlabel('Scene', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ssim_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. LPIPS comparison
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='scene', y='lpips', hue='experiment')
    plt.title('LPIPS Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('LPIPS', fontsize=12)
    plt.xlabel('Scene', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lpips_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plots for distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.boxplot(data=df, x='experiment', y='psnr', ax=axes[0])
    axes[0].set_title('PSNR Distribution', fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    sns.boxplot(data=df, x='experiment', y='ssim', ax=axes[1])
    axes[1].set_title('SSIM Distribution', fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    sns.boxplot(data=df, x='experiment', y='lpips', ax=axes[2])
    axes[2].set_title('LPIPS Distribution', fontweight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {output_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray Sampling Research Experiments")
    parser.add_argument("--mode", choices=["quick", "primary", "secondary", "ablation", "all"],
                       default="quick", help="Experiment mode")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis")
    
    args = parser.parse_args()
    
    # Select experiments based on mode
    if args.mode == "quick":
        experiments = {k: PRIMARY_EXPERIMENTS[k] for k in QUICK_TEST["experiments"]}
        datasets = {QUICK_TEST["dataset"]: {
            **DATASETS[QUICK_TEST["dataset"]],
            "scenes": QUICK_TEST["scenes"],
            "iterations": QUICK_TEST["iterations"],
        }}
    elif args.mode == "primary":
        experiments = PRIMARY_EXPERIMENTS
        datasets = DATASETS
    elif args.mode == "secondary":
        experiments = {**PRIMARY_EXPERIMENTS, **SECONDARY_EXPERIMENTS}
        datasets = DATASETS
    elif args.mode == "ablation":
        experiments = {**ABLATION_BLUR, **ABLATION_KERNEL, **ABLATION_SAMPLES}
        datasets = {"blender": {**DATASETS["blender"], "scenes": ["lego"]}}
    else:  # all
        experiments = {
            **PRIMARY_EXPERIMENTS,
            **SECONDARY_EXPERIMENTS,
            **ABLATION_BLUR,
            **ABLATION_KERNEL,
            **ABLATION_SAMPLES,
        }
        datasets = DATASETS
    
    print(f"\n{'='*80}")
    print(f"🎯 RAY SAMPLING RESEARCH - {args.mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"Experiments: {len(experiments)}")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Total runs: {sum(len(d['scenes']) for d in datasets.values()) * len(experiments)}")
    print(f"{'='*80}\n")
    
    if not args.eval_only and not args.analyze_only:
        # Run training
        print("\n🚀 STAGE 1: TRAINING")
        print("="*80)
        run_experiment_suite(experiments, datasets)
    
    if not args.analyze_only:
        # Run evaluation
        print("\n📊 STAGE 2: EVALUATION")
        print("="*80)
        df = evaluate_all_experiments(experiments, datasets)
    else:
        # Load existing metrics
        df = pd.read_csv("all_metrics.csv")
    
    # Analysis
    print("\n📈 STAGE 3: ANALYSIS")
    print("="*80)
    
    # Summary statistics
    summary = compute_summary_statistics(df)
    print("\nSummary Statistics:")
    print(summary)
    
    # Statistical significance
    sig_tests = compute_statistical_significance(df)
    print("\nStatistical Significance Tests:")
    print(sig_tests)
    
    # Generate LaTeX table
    generate_latex_table(df)
    
    # Generate plots
    plot_results(df)
    
    print("\n✅ ALL DONE!")
    print("="*80)
    print("Results:")
    print("  - Metrics: all_metrics.csv")
    print("  - LaTeX table: results_table.tex")
    print("  - Plots: plots/")
    print("="*80)


if __name__ == "__main__":
    main()
