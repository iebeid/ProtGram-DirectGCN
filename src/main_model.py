import mmap
import os
import collections
sequence_file = "C:/ProgramData/ProtDiGCN/sequence_sample.txt"
sequence_bigrams = []
concatenated_bigrams_for_counting = []
vocab = []
try:
    with open(sequence_file, "rb") as f:
        print(f"\nFile '{sequence_file}' opened for reading.")
        fileno = f.fileno()
        file_size = os.fstat(fileno).st_size
        print(f"File size: {file_size} bytes")
        if file_size == 0:
            print("File is empty, nothing to map.")
            exit()
        with mmap.mmap(fileno, length=0, access=mmap.ACCESS_READ) as mm:
            print("Memory map created successfully.")
            print(f"Mapped object type: {type(mm)}")
            print(f"Mapped object length: {len(mm)} bytes (should match file size)")
            current_pos = 0
            for i in range(5):
                next_newline = mm.find(b'\n', current_pos)
                if next_newline == -1:
                    if current_pos < len(mm):  # Last line
                        line_bytes = mm[current_pos:]
                        line_decoded = line_bytes.decode('utf-8', errors='replace')
                        print(f"Line {i + 1}: {line_decoded}")
                    break
                line_bytes = mm[current_pos:next_newline]
                line_decoded = line_bytes.decode('utf-8', errors='replace')
                length_of_sequences = len(line_decoded)
                # Extract bigrams
                for i in range(length_of_sequences):
                    if i == length_of_sequences:
                        break
                    current_residue = line_decoded[i]
                    next_residue = line_decoded[i+1]
                    vocab.append(str(ord(current_residue)))
                    sequence_bigrams.append((current_residue, next_residue))
                    concatenated_bigrams_for_counting.append(str(ord(current_residue))+"|"+str(ord(next_residue)))
                ##################
                print(f"Line {i + 1}: {line_decoded}")
                current_pos = next_newline + 1
                if current_pos >= len(mm):
                    break
            print("\nMemory map will be closed automatically.")
        print(f"File '{sequence_file}' closed automatically.")
except FileNotFoundError:
    print(f"Error: File '{sequence_file}' not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    print(f"\nScript finished. You can inspect '{sequence_file}'.")
sequence_bigrams = sequence_bigrams[0:len(sequence_bigrams)-1] # remove final edge
bigram_counts = collections.Counter(concatenated_bigrams_for_counting)
vocab = list(set(vocab))
print(vocab)
total_transitions = float(sum(bigram_counts.values()))
bigram_frequencies = {}
for edge, count in bigram_counts.items():
    frequency = count / total_transitions
    bigram_frequencies[edge] = frequency
print(sequence_bigrams)
print(len(sequence_bigrams))
print(bigram_counts)
print(bigram_frequencies)




# 1- enter the size of the context window

# Repeat as the size of the context window starting with 1:
### 2- compute unique nodes
### 3- compute unique counts
### 4- compute unique frequencies
### 5- compute edges (dict; keys are unique nodes and values are 2 lists one in nodes and out nodes)
### 6- compute unique edge counts
### 7- compute unique edge frequencies
### 8- create directed graph (pass nodes and edges as list of tuples with 3 cells) and add to list of graph objects

# Repeat for all graphs in list of graph objects
### 9- get current graph object
### 10- initialize features from previous embedding matrix
### 11- train DiGCN to predict next node in the graph
### 12- extract embedding

# final embedding will be let's say per trigram so the embedding will have to be lookedup and pooled per protein


