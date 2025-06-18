import dask.bag as db
import os

def tokenize_text(text_line):
    """
    Simple tokenization function: splits a line of text into words.
    You can replace this with a more advanced tokenization method
    (e.g., using NLTK or spaCy) for real-world applications.
    """
    return text_line.strip().lower().split()

def process_text_file_with_dask(file_path, num_partitions=None):
    """
    Reads a text file and tokenizes its content in parallel using Dask.

    Args:
        file_path (str): The path to the text file.
        num_partitions (int, optional): The number of partitions to split
                                       the file into. If None, Dask will
                                       determine an appropriate number.
                                       This relates to the number of parallel
                                       tasks.

    Returns:
        list: A list of lists, where each inner list contains the tokens
              from a line of the input file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Create a Dask Bag from the text file.
    # Each line of the file becomes an item in the bag.
    # We can specify 'num_partitions' to control the level of parallelism.
    text_bag = db.read_text(file_path, blocksize=None).repartition(npartitions=num_partitions)


    # Map the tokenization function over each line in the Dask Bag.
    # Dask will execute this in parallel across threads (or processes/cluster).
    tokenized_bag = text_bag.map(tokenize_text)

    # Compute the result. This triggers the parallel execution.
    all_tokens = tokenized_bag.compute()

    return all_tokens

if __name__ == "__main__":
    # 1. Create a dummy text file for demonstration
    dummy_file_name = "sample_text.txt"
    sample_content = (
        "This is the first line of text.\n"
        "Dask makes parallel processing easy.\n"
        "Another line for tokenization demonstration.\n"
        "Hello World from Dask."
    )
    with open(dummy_file_name, "w") as f:
        f.write(sample_content)

    print(f"Created dummy file: '{dummy_file_name}'\n")

    # 2. Process the dummy file with Dask
    try:
        # You can adjust num_partitions based on your file size and system resources
        # If set to None, Dask will estimate.
        # Setting a small number like 2 or 4 is good for testing parallelism.
        tokenized_output = process_text_file_with_dask(dummy_file_name, num_partitions=2)

        print("Tokenization complete. Here are the first few tokenized lines:")
        for i, tokens in enumerate(tokenized_output):
            print(f"Line {i+1}: {tokens}")
            if i >= 4: # Print only first 5 lines for brevity
                break

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists(dummy_file_name):
            os.remove(dummy_file_name)
            print(f"\nCleaned up dummy file: '{dummy_file_name}'")