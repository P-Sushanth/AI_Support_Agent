from src.analysis.failures import run_failure_analysis

if __name__ == "__main__":
    failures = run_failure_analysis()
    print(f"Failure analysis execution complete. Identified {len(failures)} failures.")
