def filter_log_file(input_file, output_file):
    keep_patterns = ("{'task':", "Postprocessed")

    with open(input_file, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if line.startswith(keep_patterns)]

    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)

input_file = 'log_bkk_today copy.txt'
output_file = 'filtered_log.txt'

filter_log_file(input_file, output_file)
