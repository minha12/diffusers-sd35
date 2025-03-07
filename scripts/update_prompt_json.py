import json
import os
from datetime import datetime
from pathlib import Path

# Get user's home directory and construct paths
HOME = Path.home()
DRSK_DIR = HOME / "ControlNet/training/drsk"
PROMPT_JSON_PATH = DRSK_DIR / "prompt.json"
BACKUP_PATH = DRSK_DIR / f"prompt.json.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

def transform_source_path(source_path):
    """Transform the source path by replacing '_mask.png' with '.png'"""
    return source_path.replace('_mask.png', '.png')

def main():
    # Check if the source file exists
    if not PROMPT_JSON_PATH.exists():
        print(f"Error: Cannot find {PROMPT_JSON_PATH}")
        return

    # Create backup of original file
    try:
        print(f"Creating backup at {BACKUP_PATH}")
        with open(PROMPT_JSON_PATH, 'r') as src, open(BACKUP_PATH, 'w') as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"Error creating backup: {e}")
        return

    try:
        # Read the original prompt.json
        with open(PROMPT_JSON_PATH, 'r') as f:
            drsk_data = [json.loads(line) for line in f]

        # Create new data structure
        updated_data = []
        for item in drsk_data:
            new_item = {
                "source": f"source/{transform_source_path(item['source'])}", 
                "target": f"target/{item['target']}", 
                "prompt": item['prompt']
            }
            updated_data.append(new_item)

        # Write the updated data back to prompt.json
        with open(PROMPT_JSON_PATH, 'w') as f:
            for item in updated_data:
                json.dump(item, f)
                f.write('\n')

        print(f"Successfully updated {PROMPT_JSON_PATH}")
        print(f"Backup saved at {BACKUP_PATH}")
        
        # Print example of transformation for verification
        if updated_data:
            print("\nExample of transformation:")
            print(f"Original source: {drsk_data[0]['source']}")
            print(f"Updated source: {updated_data[0]['source']}")

    except Exception as e:
        print(f"Error processing file: {e}")
        # Try to restore from backup if something went wrong
        try:
            with open(BACKUP_PATH, 'r') as src, open(PROMPT_JSON_PATH, 'w') as dst:
                dst.write(src.read())
            print("Restored from backup due to error")
        except Exception as restore_error:
            print(f"Error restoring from backup: {restore_error}")

if __name__ == "__main__":
    main()