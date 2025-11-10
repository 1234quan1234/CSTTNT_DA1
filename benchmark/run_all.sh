#!/bin/bash
# filepath: /home/bui-anh-quan/CSTTNT_DA1/benchmark/run_all.sh

set -e  # Exit on error (disabled for individual benchmark runs)

# ============================================================================
# MASTER BENCHMARK RUNNER
# Runs all benchmarks sequentially with logging and error handling
# ============================================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/benchmark_run_${TIMESTAMP}.log"

# Trap Ctrl+C for graceful cleanup
trap cleanup INT TERM

cleanup() {
    echo -e "\n${YELLOW}⚠ Interrupted by user. Cleaning up...${NC}"
    if [ -f "$TEMP_FILE" ]; then
        rm -f "$TEMP_FILE"
    fi
    echo -e "${YELLOW}✓ Cleanup complete. Exiting.${NC}"
    exit 130
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [$level] $message" >> "$MAIN_LOG"
    
    case $level in
        ERROR)
            echo -e "${RED}✗ $message${NC}"
            ;;
        SUCCESS)
            echo -e "${GREEN}✓ $message${NC}"
            ;;
        WARNING)
            echo -e "${YELLOW}⚠ $message${NC}"
            ;;
        INFO)
            echo -e "${CYAN}ℹ $message${NC}"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

print_header() {
    local title="$1"
    local width=80
    echo -e "\n${BOLD}${BLUE}$(printf '=%.0s' $(seq 1 $width))${NC}"
    echo -e "${BOLD}${BLUE}$title${NC}"
    echo -e "${BOLD}${BLUE}$(printf '=%.0s' $(seq 1 $width))${NC}\n"
}

print_progress() {
    local current=$1
    local total=$2
    local task=$3
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    printf "\r${CYAN}[%s%s] %3d%% - %s${NC}" \
        "$(printf '#%.0s' $(seq 1 $filled))" \
        "$(printf ' %.0s' $(seq 1 $empty))" \
        "$percent" \
        "$task"
}

check_python_env() {
    log INFO "Checking Python environment..."
    
    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        log ERROR "Python 3 not found. Please install Python 3.7+"
        exit 1
    fi
    
    local python_version=$(python3 --version | awk '{print $2}')
    log SUCCESS "Python $python_version found"
    
    # Check virtual environment (optional but recommended)
    if [ -z "$VIRTUAL_ENV" ]; then
        log WARNING "No virtual environment detected. Consider using venv."
    else
        log SUCCESS "Virtual environment active: $VIRTUAL_ENV"
    fi
    
    return 0
}

check_dependencies() {
    log INFO "Checking dependencies..."
    
    local required_packages=("numpy" "scipy" "pandas" "matplotlib" "seaborn")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log ERROR "Missing packages: ${missing_packages[*]}"
        log INFO "Install with: pip install ${missing_packages[*]}"
        return 1
    fi
    
    log SUCCESS "All dependencies installed"
    return 0
}

setup_directories() {
    log INFO "Setting up directories..."
    
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$RESULTS_DIR/rastrigin"
    mkdir -p "$RESULTS_DIR/knapsack"
    mkdir -p "$RESULTS_DIR/plots"
    
    log SUCCESS "Directories created"
}

estimate_time() {
    local scenario=$1
    
    case $scenario in
        "rastrigin_quick")
            echo "~3 minutes"
            ;;
        "rastrigin_multimodal")
            echo "~5 minutes"
            ;;
        "rastrigin_scalability")
            echo "~8 minutes"
            ;;
        "knapsack_small")
            echo "~15 minutes"
            ;;
        "knapsack_medium")
            echo "~30 minutes"
            ;;
        "knapsack_large")
            echo "~45 minutes"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

run_benchmark() {
    local script=$1
    local args=$2
    local scenario_name=$3
    local current=$4
    local total=$5
    
    local start_time=$(date +%s)
    local eta=$(estimate_time "$scenario_name")
    
    print_progress $current $total "$scenario_name (ETA: $eta)"
    
    log INFO "Starting: $scenario_name"
    log INFO "Command: python3 $script $args"
    
    # Run benchmark and capture output
    local output_log="$LOG_DIR/${scenario_name}_${TIMESTAMP}.log"
    
    if python3 "$script" $args > "$output_log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        echo "" # New line after progress bar
        log SUCCESS "$scenario_name completed in ${minutes}m ${seconds}s"
        return 0
    else
        local exit_code=$?
        echo "" # New line after progress bar
        log ERROR "$scenario_name failed with exit code $exit_code"
        log ERROR "Check log: $output_log"
        return 1
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
        local start_time=$(date +%s)
    local failed_scenarios=()
    local successful_scenarios=()
    
    print_header "BENCHMARK SUITE - MASTER RUNNER"
    
    log INFO "Starting benchmark run at $(date)"
    log INFO "Results directory: $RESULTS_DIR"
    log INFO "Log file: $MAIN_LOG"
    
    # Pre-flight checks
    check_python_env || exit 1
    check_dependencies || exit 1
    setup_directories
    
    # Define scenarios - FIX: Added --jobs -1 and proper args
    declare -a scenarios=(
        "rastrigin:--config quick_convergence --jobs -1:rastrigin_quick"
        "rastrigin:--config multimodal_escape --jobs -1:rastrigin_multimodal"
        "rastrigin:--config scalability --jobs -1:rastrigin_scalability"
        "knapsack:--size 50 --jobs -1:knapsack_small"
        "knapsack:--size 100 --jobs -1:knapsack_medium"
        "knapsack:--size 200 --jobs -1:knapsack_large"
    )
    
    local total_scenarios=${#scenarios[@]}
    local current=0
    
    print_header "RUNNING BENCHMARKS (${total_scenarios} scenarios)"
    
    # Run each scenario
    for scenario in "${scenarios[@]}"; do
        IFS=':' read -r problem args name <<< "$scenario"
        current=$((current + 1))
        
        local script="$SCRIPT_DIR/run_${problem}.py"
        
        if [ ! -f "$script" ]; then
            log ERROR "Script not found: $script"
            failed_scenarios+=("$name")
            continue
        fi
        
        # Run benchmark
        if run_benchmark "$script" "$args" "$name" "$current" "$total_scenarios"; then
            successful_scenarios+=("$name")
        else
            failed_scenarios+=("$name")
            log WARNING "Continuing with next scenario..."
        fi
        
        echo "" # Spacing between scenarios
    done
    
    # Analysis and visualization
    print_header "GENERATING ANALYSIS AND VISUALIZATIONS"
    
    log INFO "Running statistical analysis..."
    if python3 "$SCRIPT_DIR/analyze_results.py" \
        --problem all \
        --rastrigin-dir "$RESULTS_DIR/rastrigin" \
        --knapsack-dir "$RESULTS_DIR/knapsack" \
        --output-dir "$RESULTS_DIR" > "$LOG_DIR/analysis_${TIMESTAMP}.log" 2>&1; then
        log SUCCESS "Statistical analysis completed"
    else
        log ERROR "Statistical analysis failed"
    fi
    
    log INFO "Generating visualizations..."
    if python3 "$SCRIPT_DIR/visualize.py" \
        --rastrigin-dir "$RESULTS_DIR/rastrigin" \
        --knapsack-dir "$RESULTS_DIR/knapsack" \
        --output-dir "$RESULTS_DIR/plots" > "$LOG_DIR/visualize_${TIMESTAMP}.log" 2>&1; then
        log SUCCESS "Visualizations generated"
    else
        log ERROR "Visualization generation failed"
    fi
    
    # Generate summary report
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))
    local seconds=$((total_duration % 60))
    
    # Create summary file
    local summary_file="$RESULTS_DIR/benchmark_summary_${TIMESTAMP}.txt"
    
    {
        echo "BENCHMARK RUN SUMMARY"
        echo "===================="
        echo ""
        echo "Timestamp: $(date)"
        echo "Total duration: ${hours}h ${minutes}m ${seconds}s"
        echo ""
        echo "Scenarios run: ${total_scenarios}"
        echo "Successful: ${#successful_scenarios[@]}"
        echo "Failed: ${#failed_scenarios[@]}"
        echo ""
        
        if [ ${#successful_scenarios[@]} -gt 0 ]; then
            echo "Successful scenarios:"
            for scenario in "${successful_scenarios[@]}"; do
                echo "  ✓ $scenario"
            done
            echo ""
        fi
        
        if [ ${#failed_scenarios[@]} -gt 0 ]; then
            echo "Failed scenarios:"
            for scenario in "${failed_scenarios[@]}"; do
                echo "  ✗ $scenario"
            done
            echo ""
        fi
        
        echo "Results location: $RESULTS_DIR"
        echo "Logs location: $LOG_DIR"
        echo "Main log: $MAIN_LOG"
        
    } > "$summary_file"
    
    # Print final summary
    print_header "BENCHMARK RUN COMPLETE"
    
    cat "$summary_file"
    
    log INFO "Summary saved to: $summary_file"
    
    if [ ${#failed_scenarios[@]} -eq 0 ]; then
        log SUCCESS "All benchmarks completed successfully!"
        exit 0
    else
        log WARNING "${#failed_scenarios[@]} scenario(s) failed. Check logs for details."
        exit 1
    fi
}

# Run main function
cd "$PROJECT_ROOT"
main "$@"