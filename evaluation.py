import pandas as pd
import subprocess

# Function to extract response from subprocess output
def extract_response(text):
    marker = "ASSISTANT:"
    start_index = text.find(marker)
    if start_index != -1:
        return text[start_index + len(marker):].strip()
    else:
        return None

# Specify the file path
file_path = 'MAPQADataset.xlsx'

# Read the Excel file
all_sheets = pd.read_excel(file_path, sheet_name=None)
sheet = all_sheets['Sheet1']

results = []

try:
    count = 0
    # Iterate through sheets and rows
    for sheet_name, df in all_sheets.items():
        count += 1
        print(f"Data from sheet: {sheet_name}")

        for index, row in df.iterrows():
            try:
                # Extract data from current row
                ques = row[0]
                snap_id = f"Vdata/{row[2]}.PNG"
                options = row[3]
                ans = str(int(row[4]))

                # Prepare prompt for subprocess
                prompt = f"The question is: {ques}. Here are the options: {options}. Just choose the correct option from the image also show the answer with visible quotation and give proper explanation."
                
                print(f"Processing row {index}: {prompt} with image {snap_id}")

                # Call the secondary script using subprocess
                result = subprocess.run(['python', '<model_name>.py', '--prompt', prompt, '--image_url', snap_id], capture_output=True, text=True)
                
                if result.returncode == 0:
                    answer = extract_response(result.stdout.strip())
                    
                    print(f"Response: {answer}")

                    # Collect data for the results list
                    results.append({
                        'ques': ques,
                        'sheet_name': sheet_name,
                        'row': index,
                        'image': snap_id,
                        'question': ques,
                        'options': options,
                        'correct_ans': ans,
                        'response': answer
                    })
                
                else:
                    print(f"Error: {result.stderr}")
                
            except Exception as e:
                print(f"Error processing the data: {e}")

except Exception as e:
    print(f"Error {e}")

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
output_file = '<model_name>.xlsx'
results_df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")
