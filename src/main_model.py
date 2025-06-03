import os
if "name" == "__main__":
    sequence_file = "C:/ProgramData/ProtDiGCN/sequence_sample.txt"
    with open(sequence_file, "rb") as f:
        print(f"\nFile '{sequence_file}' opened for reading.")
        fileno = f.fileno()
        file_size = os.fstat(fileno).st_size
        print(f"File size: {file_size} bytes")