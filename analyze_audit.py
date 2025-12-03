import json
import pandas as pd
import os

OUTPUT_DIR = "llm_allergen_audit_data"

def analyze_audit():
    # Find the most recent results file
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('full_audit_results_')]
    if not files:
        print(f"ERROR: No results files found in ./{OUTPUT_DIR}/")
        return
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(OUTPUT_DIR, latest_file)
    
    print(f"Loading data from: {filepath}")
    
    with open(filepath) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print("=" * 80)
    print("LLM ALLERGEN AUDIT - QUICK ANALYSIS")
    print("=" * 80)

    # Overall compliance
    total = len(df)
    passes = (df['verdict'] == 'PASS').sum()
    fails = total - passes
    print(f"\nOverall: {passes}/{total} = {passes/total*100:.1f}% pass rate")

    # By system prompt (FIXED: Use crosstab with fill_value)
    print("\n" + "-" * 80)
    print("BY SYSTEM PROMPT")
    print("-" * 80)
    
    # FIX: Use crosstab with normalize='index' to include all categories
    prompt_stats = pd.crosstab(df['system_prompt_type'], df['verdict'], normalize='index') * 100
    print(prompt_stats.round(1))

    # Hard refusal failures (the smoking gun)
    hard_failures = df[(df['system_prompt_type'] == 'hard_refusal') & (df['verdict'] == 'FAIL')]
    total_hard = len(df[df['system_prompt_type'] == 'hard_refusal'])
    print(f"\nHard Refusal Failures: {len(hard_failures)} out of {total_hard} attempts")
    if total_hard > 0:
        print(f"Hard Refusal Pass Rate: {(total_hard - len(hard_failures))/total_hard*100:.1f}%")
    else:
        print("Hard Refusal Pass Rate: N/A (no hard refusal tests in this dataset)")
    
    if len(hard_failures) > 0:
        print("\nFailure types in hard_refusal mode:")
        print(hard_failures['response_type'].value_counts())

    # Search impact (FIXED: Same crosstab approach)
    print("\n" + "-" * 80)
    print("SEARCH IMPACT")
    print("-" * 80)
    
    search_stats = pd.crosstab(
        df['search_enabled'], 
        df['system_prompt_type'], 
        values=df['verdict'], 
        aggfunc=lambda x: (x == 'PASS').mean() * 100
    )
    print(search_stats.round(1))

    # Model performance (Hard Refusal Only)
    print("\n" + "-" * 80)
    print("MODEL PERFORMANCE (Hard Refusal Only)")
    print("-" * 80)
    hard_df = df[df['system_prompt_type'] == 'hard_refusal']
    if len(hard_df) > 0:
        model_stats = hard_df.groupby('model_id')['verdict'].apply(lambda x: (x == 'PASS').mean() * 100)
        print(model_stats.sort_values(ascending=False).round(1))
    else:
        print("No hard refusal tests in this dataset")

    # Failure examples
    print("\n" + "-" * 80)
    print("EXAMPLE FAILURES (Hard Refusal)")
    print("-" * 80)
    examples = hard_failures[['model_id', 'response_type', 'failure_reason', 'response_snippet']].head(5)
    for idx, row in examples.iterrows():
        print(f"\n{row['model_id']} - {row['response_type']}:")
        print(f"  Reason: {row['failure_reason']}")
        print(f"  Response: {row['response_snippet']}")

    # Token analysis
    print("\n" + "-" * 80)
    print("TOKEN USAGE (avg per query)")
    print("-" * 80)
    print(df.groupby('model_id')['total_tokens'].mean().sort_values().round(0))

    # Cost analysis - Actual vs Estimated
    print("\n" + "-" * 80)
    print("COST ANALYSIS")
    print("-" * 80)
    total_actual = df['actual_cost'].sum()
    total_estimated = df['estimated_cost'].sum()
    cost_difference = total_actual - total_estimated
    cost_accuracy = (total_actual / total_estimated * 100) if total_estimated > 0 else 0

    print(f"Total Actual Cost: ${total_actual:.4f}")
    print(f"Total Estimated Cost: ${total_estimated:.4f}")
    print(f"Cost Difference: ${cost_difference:.4f} ({cost_accuracy:.1f}% of estimated)")
    print(f"Cost Accuracy: {'Over-estimated' if cost_difference < 0 else 'Under-estimated'} by ${abs(cost_difference):.4f}")

    # Cache utilization
    print("\n" + "-" * 80)
    print("CACHE UTILIZATION")
    print("-" * 80)
    cache_stats = df.groupby('model_id').agg({
        'cached_tokens': ['mean', 'sum'],
        'prompt_tokens': 'mean'
    }).round(0)
    cache_stats.columns = ['avg_cached', 'total_cached', 'avg_prompt']
    cache_stats['cache_ratio'] = (cache_stats['avg_cached'] / cache_stats['avg_prompt'] * 100).round(1)
    print(cache_stats.sort_values('cache_ratio', ascending=False))

    # Reasoning tokens (if any)
    if df['reasoning_tokens'].sum() > 0:
        print("\n" + "-" * 80)
        print("REASONING TOKENS")
        print("-" * 80)
        reasoning_stats = df[df['reasoning_tokens'] > 0].groupby('model_id')['reasoning_tokens'].agg(['mean', 'sum']).round(0)
        print(reasoning_stats)

    # Save summary to file
    summary_file = os.path.join(OUTPUT_DIR, f"analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Analysis of {latest_file}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall pass rate: {passes}/{total} = {passes/total*100:.1f}%\n")
        if total_hard > 0:
            f.write(f"Hard refusal pass rate: {(total_hard - len(hard_failures))/total_hard*100:.1f}% "
                    f"({total_hard - len(hard_failures)}/{total_hard})\n")
        else:
            f.write("Hard refusal pass rate: N/A (no hard refusal tests in this dataset)\n")
        if total_hard > 0:
            f.write(f"Hard refusal failures: {len(hard_failures)} ({len(hard_failures)/total_hard*100:.1f}%)\n")
        else:
            f.write("Hard refusal failures: N/A (no hard refusal tests in this dataset)\n")
        f.write(f"\nTotal Actual Cost: ${total_actual:.4f}\n")
        f.write(f"Total Estimated Cost: ${total_estimated:.4f}\n")
        f.write(f"Cost Difference: ${cost_difference:.4f} ({cost_accuracy:.1f}% of estimated)\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Prompt stats by type:\n")
        f.write(str(prompt_stats.round(1)))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Cache Utilization by Model:\n")
        f.write(str(cache_stats))
    
    print(f"\n" + "=" * 80)
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    analyze_audit()