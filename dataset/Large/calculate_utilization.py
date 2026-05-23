import json
import re

def parse_schedule_from_results(results_filepath):
    """
    Reads the results text file and extracts the production schedule and PM flags.
    Returns: { period: { line: {'pm_done': bool, 'products': [(product_id, quantity), ...]} } }
    """
    schedule = {}
    current_period = None

    with open(results_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Detect Period Header
            period_match = re.search(r'\[ PERIOD (\d+) \]', line)
            if period_match:
                current_period = int(period_match.group(1))
                schedule[current_period] = {}
                continue
            
            # 2. Detect Line Actions
            line_match = re.search(r'Line (\d+):\s+(.*)', line)
            if line_match and current_period is not None:
                line_idx = int(line_match.group(1))
                actions_str = line_match.group(2)
                
                # Check for Preventive Maintenance Flag
                pm_done = "🔧 PM" in actions_str
                
                # Check for Idle status (without PM)
                if "IDLE" in actions_str and not pm_done:
                    schedule[current_period][line_idx] = {'pm_done': False, 'products': []}
                    continue
                
                # Extract Product Quantities: P4(Q:38) -> ('4', '38')
                items = re.findall(r'P(\d+)\(Q:(\d+)\)', actions_str)
                
                schedule[current_period][line_idx] = {
                    'pm_done': pm_done,
                    'products': [(int(p), int(q)) for p, q in items]
                }
                
    return schedule

def calculate_exact_capacity_utilization(json_filepath, schedule):
    """
    Calculates capacity utilization considering sequence setups, 
    processing times, expected CM (Corrective Maintenance), and PM time.
    """
    # 1. Load the parameters from the JSON file
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    num_lines = data['num_lines']
    num_periods = data['num_periods']
    proc_matrix = data['processing_time_matrix']
    setup_matrix = data['setup_time_matrix']
    first_setup = data['first_setup_time']
    capacity_per_line = data['capacity_per_line']
    
    # Maintenance parameters
    hazard_rate = data['hazard_rate']
    cm_time = data['cm_time']
    pm_time = data['pm_time']

    # Track line states for sequence setups
    last_product_on_line = {line: None for line in range(num_lines)}
    
    # Trackers
    tot_proc_time = 0.0
    tot_setup_time = 0.0
    tot_cm_time = 0.0
    tot_pm_time = 0.0

    # 2. Iterate through the schedule chronologically
    for period in sorted(schedule.keys()):
        for line in sorted(schedule[period].keys()):
            day_plan = schedule[period][line]
            
            # --- A. PREVENTIVE MAINTENANCE ---
            if day_plan['pm_done']:
                tot_pm_time += pm_time[line]
            
            # --- B. PRODUCTION & CORRECTIVE MAINTENANCE & SETUP ---
            for product, quantity in day_plan['products']:
                
                # 1. Processing Time
                p_time = quantity * proc_matrix[line][product]
                tot_proc_time += p_time
                
                # 2. Expected Corrective Maintenance Time
                c_time = hazard_rate[line] * cm_time[line] * p_time
                tot_cm_time += c_time
                
                # 3. Setup Time
                prev_product = last_product_on_line[line]
                if prev_product is None:
                    # Cold start
                    s_time = first_setup[line]
                elif prev_product != product:
                    # Sequence changeover
                    s_time = setup_matrix[line][prev_product][product]
                else:
                    # Same product continuation
                    s_time = 0.0
                    
                tot_setup_time += s_time
                
                # Update line state
                last_product_on_line[line] = product

    # 3. Aggregate 
    total_time_used = tot_proc_time + tot_setup_time + tot_cm_time + tot_pm_time
    total_capacity = sum(capacity_per_line) * num_periods
    utilization_pct = (total_time_used / total_capacity) * 100
    
    return {
        "Total Capacity (hrs)": total_capacity,
        "Processing Time (hrs)": round(tot_proc_time, 2),
        "Setup Time (hrs)": round(tot_setup_time, 2),
        "Expected CM Time (hrs)": round(tot_cm_time, 2),
        "PM Time (hrs)": round(tot_pm_time, 2),
        "Total Time Used (hrs)": round(total_time_used, 2),
        "Utilization (%)": round(utilization_pct, 2)
    }

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # Simply point these to the respective files
    RESULTS_FILE = "instance_5_2_4_results.txt"
    JSON_FILE = "instance_5_2_4.json"
    
    try:
        # 1. Parse text schedule
        print(f"📄 Parsing schedule from '{RESULTS_FILE}'...")
        auto_schedule = parse_schedule_from_results(RESULTS_FILE)
        
        # 2. Calculate Exact Capacity logic
        print(f"🧮 Loading parameters from '{JSON_FILE}' and computing MILP capacity rules...\n")
        results = calculate_exact_capacity_utilization(JSON_FILE, auto_schedule)
        
        # 3. Display Detailed Output
        print("🏆 DETAILED CAPACITY UTILIZATION REPORT")
        print("="*45)
        print(f"🏭 Total Capacity Available : {results['Total Capacity (hrs)']:>7} hrs")
        print("-" * 45)
        print(f"⚙️  Processing Time          : {results['Processing Time (hrs)']:>7} hrs")
        print(f"🔧 Setup Time               : {results['Setup Time (hrs)']:>7} hrs")
        print(f"⚠️  Expected CM Time         : {results['Expected CM Time (hrs)']:>7} hrs")
        print(f"🛡️  Preventive Maint (PM)    : {results['PM Time (hrs)']:>7} hrs")
        print("-" * 45)
        print(f"⏱️  TOTAL HOURS CONSUMED     : {results['Total Time Used (hrs)']:>7} hrs")
        print(f"📈 CAPACITY UTILIZATION     : {results['Utilization (%)']:>7} %")
        print("="*45)
        
    except FileNotFoundError as e:
        print(f"⚠️ Error: Could not find file -> {e.filename}")