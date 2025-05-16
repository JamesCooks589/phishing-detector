import subprocess
import os
import sys
import time
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle any errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stderr)
        return False

def check_requirements():
    """Check and install required packages"""
    print("Checking and installing requirements...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print("Error installing requirements:")
        print(e.stderr)
        return False

def main():
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    
    # Define the execution order
    steps = [
        ("data_preprocessing.py", "Preprocessing data and creating TF-IDF vectorizer"),
        ("train_model.py", "Training the Multinomial Naive Bayes model"),
        ("explain_model.py", "Generating model explanations")
    ]
    
    # Check requirements first
    if not check_requirements():
        print("Failed to install requirements. Exiting.")
        sys.exit(1)
    
    # Run each script in order
    for script, description in steps:
        if not run_script(script, description):
            print(f"\nError in {script}. Stopping execution.")
            sys.exit(1)
        time.sleep(1)  # Small delay between scripts
    
    print("\n✅ All preprocessing and training complete!")
    print("\nStarting Streamlit application...")
    
    # Run the Streamlit app
    try:
        print("\nTo access the application, open your web browser to the URL shown below:")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("Error starting Streamlit application:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
