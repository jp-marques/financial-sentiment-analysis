import os
import json
import base64

def extract_saved_images():
    """
    Reads the .ipynb file directly, finds the saved image outputs,
    and saves them to the assets directory.
    
    This avoids environment issues with nbconvert by not re-executing the notebook.
    """
    notebook_path = os.path.join("notebooks", "04_visualizations.ipynb")
    output_dir = os.path.join(".github", "assets")
    
    print(f"Reading notebook: {notebook_path}")

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_json = json.load(f)
            
        print("Notebook read successfully. Now extracting saved images...")
        
        image_count = 0
        # This mapping remains the same, as it's based on the notebook's structure
        cell_to_filename_map = {
            8: "dashboard.png",
            12: "kpis.png",
            14: "roi.png"
        }

        # Iterate through the cells
        for i, cell in enumerate(notebook_json['cells']):
            cell_index_from_1 = i + 1 # Notebooks are often discussed in 1-based index
            
            # We are mapping to the visual cell number. In the notebook file, this is index-1
            actual_index = i

            if actual_index in cell_to_filename_map and 'outputs' in cell:
                for output in cell['outputs']:
                    if 'data' in output and 'image/png' in output['data']:
                        image_data = base64.b64decode(output['data']['image/png'])
                        
                        filename = cell_to_filename_map[actual_index]
                        output_path = os.path.join(output_dir, filename)
                        
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        
                        print(f"✅ Saved {filename} from cell {actual_index} to {output_dir}")
                        image_count += 1
                        break # Move to the next cell
        
        if image_count == len(cell_to_filename_map):
            print("\nAll target images extracted successfully!")
        else:
            print(f"\nWarning: Extracted {image_count} images, but expected {len(cell_to_filename_map)}. Check cell indices.")

    except FileNotFoundError:
        print(f"ERROR: The notebook file was not found at {notebook_path}")
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode the JSON from the notebook file. It may be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    extract_saved_images()
