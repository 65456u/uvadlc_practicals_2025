"""
Parser for terminal_output_20251125_221746.txt
Extracts validation and test accuracy results from the experimental runs.
"""
import re
import json
import statistics


def parse_terminal_output(filepath):
    """Parse terminal output and extract experiment results."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all experiment sections marked by the asterisk blocks
    experiment_pattern = r'\*+\n Type of convolution :  (\w+)\n Type of network :  (Net\d+)\n.*?\n\*+\nResults for validation dataset \{([^}]+)\}\nmean: ([\d.]+) std: ([\d.]+) for validation\nResults for test dataset \{([^}]+)\}\nmean: ([\d.]+) std: ([\d.]+) for test'
    
    matches = re.finditer(experiment_pattern, content, re.DOTALL)
    
    results = []
    
    for match in matches:
        conv_type = match.group(1)
        net_type = match.group(2)
        val_dict_str = match.group(3)
        val_mean = float(match.group(4))
        val_std = float(match.group(5))
        test_dict_str = match.group(6)
        test_mean = float(match.group(7))
        test_std = float(match.group(8))
        
        # Parse the dictionaries
        val_dict = {}
        test_dict = {}
        
        # Parse validation results
        for item in val_dict_str.split(','):
            key, value = item.split(':')
            val_dict[int(key.strip())] = float(value.strip())
        
        # Parse test results
        for item in test_dict_str.split(','):
            key, value = item.split(':')
            test_dict[int(key.strip())] = float(value.strip())
        
        results.append({
            'conv_type': conv_type,
            'net_type': net_type,
            'validation': {
                'mean': val_mean,
                'std': val_std,
                'per_round': val_dict
            },
            'test': {
                'mean': test_mean,
                'std': test_std,
                'per_round': test_dict
            }
        })
    
    return results


def organize_results(results):
    """Organize results by network type and convolution type."""
    organized = {
        'Net1': {},
        'Net2': {}
    }
    
    for result in results:
        net_type = result['net_type']
        conv_type = result['conv_type']
        organized[net_type][conv_type] = {
            'validation_mean': result['validation']['mean'],
            'validation_std': result['validation']['std'],
            'test_mean': result['test']['mean'],
            'test_std': result['test']['std']
        }
    
    return organized


def print_summary_tables(organized):
    """Print formatted tables for the questions."""
    
    conv_types = ['valid', 'replicate', 'reflect', 'circular', 'sconv', 'fconv']
    
    print("=" * 80)
    print("QUESTION 1.1 (a) i: Net1 Results")
    print("=" * 80)
    print(f"{'Conv Type':<12} {'Val Mean':<12} {'Val Std':<12} {'Test Mean':<12} {'Test Std':<12}")
    print("-" * 80)
    
    for conv_type in conv_types:
        if conv_type in organized['Net1']:
            data = organized['Net1'][conv_type]
            print(f"{conv_type:<12} {data['validation_mean']:>10.2f}% {data['validation_std']:>10.4f}% {data['test_mean']:>10.2f}% {data['test_std']:>10.4f}%")
    
    print("\n" + "=" * 80)
    print("QUESTION 1.1 (c) i: Net2 Results")
    print("=" * 80)
    print(f"{'Conv Type':<12} {'Val Mean':<12} {'Val Std':<12} {'Test Mean':<12} {'Test Std':<12}")
    print("-" * 80)
    
    for conv_type in conv_types:
        if conv_type in organized['Net2']:
            data = organized['Net2'][conv_type]
            print(f"{conv_type:<12} {data['validation_mean']:>10.2f}% {data['validation_std']:>10.4f}% {data['test_mean']:>10.2f}% {data['test_std']:>10.4f}%")
    
    print("\n" + "=" * 80)
    print("QUESTION 1.1 (c) ii: Comparison of Test Accuracy (Net2 vs Net1)")
    print("=" * 80)
    print(f"{'Conv Type':<12} {'Net1 Test':<12} {'Net2 Test':<12} {'Change':<12} {'Direction':<12}")
    print("-" * 80)
    
    for conv_type in conv_types:
        if conv_type in organized['Net1'] and conv_type in organized['Net2']:
            net1_test = organized['Net1'][conv_type]['test_mean']
            net2_test = organized['Net2'][conv_type]['test_mean']
            change = net2_test - net1_test
            direction = "Increased" if change > 0 else ("Decreased" if change < 0 else "No change")
            print(f"{conv_type:<12} {net1_test:>10.2f}% {net2_test:>10.2f}% {change:>+10.2f}% {direction:<12}")
    
    print("\n" + "=" * 80)


def save_to_files(results, organized):
    """Save results to JSON and markdown files."""
    
    # Save full results to JSON
    with open('parsed_results.json', 'w') as f:
        json.dump({
            'full_results': results,
            'organized': organized
        }, f, indent=2)
    
    print("\n✓ Results saved to parsed_results.json")
    
    # Create markdown table
    conv_types = ['valid', 'replicate', 'reflect', 'circular', 'sconv', 'fconv']
    
    with open('results_tables.md', 'w') as f:
        f.write("# Question 1.1 Results\n\n")
        
        f.write("## (a) i: Net1 Results\n\n")
        f.write("| Conv Type | Validation Mean (%) | Validation Std (%) | Test Mean (%) | Test Std (%) |\n")
        f.write("|-----------|---------------------|--------------------|--------------|--------------|\n")
        for conv_type in conv_types:
            if conv_type in organized['Net1']:
                data = organized['Net1'][conv_type]
                f.write(f"| {conv_type:<9} | {data['validation_mean']:>17.2f} | {data['validation_std']:>16.4f} | {data['test_mean']:>11.2f} | {data['test_std']:>11.4f} |\n")
        
        f.write("\n## (c) i: Net2 Results\n\n")
        f.write("| Conv Type | Validation Mean (%) | Validation Std (%) | Test Mean (%) | Test Std (%) |\n")
        f.write("|-----------|---------------------|--------------------|--------------|--------------|\n")
        for conv_type in conv_types:
            if conv_type in organized['Net2']:
                data = organized['Net2'][conv_type]
                f.write(f"| {conv_type:<9} | {data['validation_mean']:>17.2f} | {data['validation_std']:>16.4f} | {data['test_mean']:>11.2f} | {data['test_std']:>11.4f} |\n")
        
        f.write("\n## (c) ii: Comparison of Test Accuracy (Net2 vs Net1)\n\n")
        f.write("| Conv Type | Net1 Test (%) | Net2 Test (%) | Change (%) | Direction |\n")
        f.write("|-----------|---------------|---------------|------------|----------|\n")
        for conv_type in conv_types:
            if conv_type in organized['Net1'] and conv_type in organized['Net2']:
                net1_test = organized['Net1'][conv_type]['test_mean']
                net2_test = organized['Net2'][conv_type]['test_mean']
                change = net2_test - net1_test
                direction = "Increased" if change > 0 else ("Decreased" if change < 0 else "No change")
                f.write(f"| {conv_type:<9} | {net1_test:>12.2f} | {net2_test:>12.2f} | {change:>+9.2f} | {direction:<8} |\n")
    
    print("✓ Markdown tables saved to results_tables.md")


if __name__ == "__main__":
    print("Parsing terminal_output_20251125_221746.txt...")
    
    results = parse_terminal_output('terminal_output_20251125_221746.txt')
    print(f"✓ Found {len(results)} experiments")
    
    organized = organize_results(results)
    
    print_summary_tables(organized)
    
    save_to_files(results, organized)
    
    print("\n" + "=" * 80)
    print("PARSING COMPLETE!")
    print("=" * 80)
