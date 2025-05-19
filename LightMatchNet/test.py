# Read lines from both files
with open('data/VeRi/name_test.txt', 'r') as f1:
    lines1 = [line.strip() for line in f1.readlines()]

with open('data/VeRi/name_query.txt', 'r') as f2:
    lines2 = [line.strip() for line in f2.readlines()]

# Combine and count occurrences
all_lines = lines1 + lines2

# Get unique lines (appearing only once across both files)
unique_lines = [line for line in all_lines if all_lines.count(line) == 1]
unique_lines_query = [line for line in lines2 if lines2.count(line) == 1]

# Output or use as needed
print("Unique lines:")
print(f"Total unique lines: {len(unique_lines)}")
print(f"Total lines in name_query.txt: {len(lines2)}")
print(f"Total lines in name_test.txt: {len(lines1)}")
print(f"Total unique lines in name_query.txt: {len(unique_lines_query)}")
print(f"Total unique lines in name_test.txt: {len(set(lines2))}")
print(f"Total lines: {len(all_lines)}")
    
