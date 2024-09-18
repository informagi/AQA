def filter_log_file(input_file, output_file):
    # Define the start patterns we want to keep
    keep_patterns = ("{'task':", "Postprocessed")

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Filter lines that start with the specified patterns
    filtered_lines = [line for line in lines if line.startswith(keep_patterns)]

    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)

# Specify the input and output file names
input_file = 'log_bkk_today copy.txt'
output_file = 'filtered_log.txt'

# Call the function with the file paths
filter_log_file(input_file, output_file)
