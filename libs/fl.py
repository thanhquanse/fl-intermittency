import csv
import random

def create_client_matrix(filename, rows=10, columns=10, missing_client_percentage=0.1, no_adjacent_missing=True):
    # Initialize the previous row as empty to enforce the no-adjacency rule
    prev_row = [''] * columns
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        for row_idx in range(rows):
            row = ['Y'] * columns

            if row_idx >= 2:
                N_count = int(columns * missing_client_percentage)
                
                # Ensure that 'N' is placed in positions not occupied by 'N' in the previous row
                if no_adjacent_missing:
                    possible_positions = [i for i in range(columns) if prev_row[i] != 'N']
                    N_positions = random.sample(possible_positions, min(N_count, len(possible_positions)))
                else:
                    N_positions = random.sample(range(columns), N_count)
                
                for pos in N_positions:
                    row[pos] = 'N'
            
            # Write the current row to the CSV file
            writer.writerow(row)
            
            # Update prev_row for the next iteration
            prev_row = row

if __name__ == "__main__":
    create_client_matrix("client_matrix_10x10_01_noadjacency.csv", missing_client_percentage=0.1, no_adjacent_missing=True)
    create_client_matrix("client_matrix_10x10_02_noadjacency.csv", missing_client_percentage=0.2, no_adjacent_missing=True)
    create_client_matrix("client_matrix_10x10_03_noadjacency.csv", missing_client_percentage=0.3, no_adjacent_missing=True)
    create_client_matrix("client_matrix_10x10_04_noadjacency.csv", missing_client_percentage=0.4, no_adjacent_missing=True)
    create_client_matrix("client_matrix_10x10_05_noadjacency.csv", missing_client_percentage=0.5, no_adjacent_missing=True)

    create_client_matrix("client_matrix_10x10_01_2adjacency.csv", missing_client_percentage=0.1, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_02_2adjacency.csv", missing_client_percentage=0.2, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_03_2adjacency.csv", missing_client_percentage=0.3, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_04_2adjacency.csv", missing_client_percentage=0.4, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_05_2adjacency.csv", missing_client_percentage=0.5, no_adjacent_missing=False)

    create_client_matrix("client_matrix_10x10_01_3adjacency.csv", missing_client_percentage=0.1, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_02_3adjacency.csv", missing_client_percentage=0.2, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_03_3adjacency.csv", missing_client_percentage=0.3, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_04_3adjacency.csv", missing_client_percentage=0.4, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_05_3adjacency.csv", missing_client_percentage=0.5, no_adjacent_missing=False)

    create_client_matrix("client_matrix_10x10_01_4adjacency.csv", missing_client_percentage=0.1, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_02_4adjacency.csv", missing_client_percentage=0.2, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_03_4adjacency.csv", missing_client_percentage=0.3, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_04_4adjacency.csv", missing_client_percentage=0.4, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_05_4adjacency.csv", missing_client_percentage=0.5, no_adjacent_missing=False)

    create_client_matrix("client_matrix_10x10_01_5adjacency.csv", missing_client_percentage=0.1, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_02_5adjacency.csv", missing_client_percentage=0.2, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_03_5adjacency.csv", missing_client_percentage=0.3, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_04_5adjacency.csv", missing_client_percentage=0.4, no_adjacent_missing=False)
    create_client_matrix("client_matrix_10x10_05_5adjacency.csv", missing_client_percentage=0.5, no_adjacent_missing=False)