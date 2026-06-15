import os
from typing import BinaryIO
import regex as re
from collections import Counter,defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # 初始化vocab，包含所有special token和0-255的byte。
    for i, special_token in enumerate(special_tokens):
        vocab[i] = special_token.encode("utf-8")
    
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        pretoken_freq: Counter[tuple[bytes,...]] = Counter()

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_pretoken_freq: Counter[tuple[bytes,...]] = Counter()

            escaped_special_tokens = [re.escape(special_token) for special_token in special_tokens]
            splited_chunks = re.split('|'.join(escaped_special_tokens), chunk)
            for splited_chunk in splited_chunks:
                for match in re.finditer(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", splited_chunk):
                    pretoken_bytes = match.group().encode("utf-8")
                    key = tuple(pretoken_bytes[i:i+1] for i in range(len(pretoken_bytes)))
                    chunk_pretoken_freq[key] += 1
            
            pretoken_freq.update(chunk_pretoken_freq)
        
        # 根据pre-token统计byte-pair的频率，课件中提到如果每一次merge之后都重新遍历所有的pre-token效率很低，其实对于每个
        # 被merge的bytes，只有包含它的pre-token会受到影响，可以建立一个byte-pair到pre-token的索引。
        # 比如对pre-token"abcde"，本来包含了"ab","bc","cd","de"-四个bytes-pair，如果bc被merge了，那就变成了"abc","bcd","de"-三个。
        # 统计出的bytes-pair "ab","bc"和"cd"就要减去pretoken出现的频率，"abc","bcd"需要增加。
        
        bytes_pair_freq: Counter[tuple[bytes, bytes]] = Counter()
        bytes_pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes,...]]] = defaultdict(set)

        for pretoken, freq in pretoken_freq.items():
            for i in range(len(pretoken) - 1 ):
                bytes_pair_freq[pretoken[i:i+2]] += freq
                bytes_pair_to_pretokens[pretoken[i:i+2]].add(pretoken)
        
        while len(vocab) < vocab_size:
            max_freq = max(bytes_pair_freq.values())
            to_merge = max([k for k,v in bytes_pair_freq.items() if v == max_freq])

            merges.append(to_merge)
            vocab[len(vocab)] = to_merge[0] + to_merge[1]

            affected_pretokens = list(bytes_pair_to_pretokens[to_merge])
            
            for affected_pretoken in affected_pretokens:
                i = 0
                new_pretoken = []
                while i < len(affected_pretoken):
                    if affected_pretoken[i:i+2] == to_merge:
                        new_pretoken.append(to_merge[0] + to_merge[1])
                        i += 2
                    else:
                        new_pretoken.append(affected_pretoken[i])
                        i += 1
                
                new_pretoken = tuple(new_pretoken)

                pretoken_freq[new_pretoken] = pretoken_freq[affected_pretoken]
                del pretoken_freq[affected_pretoken]

                temp_freq = pretoken_freq[new_pretoken]
                for i in range(len(affected_pretoken) - 1):
                    pair = affected_pretoken[i:i+2]

                    bytes_pair_freq[pair] -= temp_freq
                    if bytes_pair_freq[pair] == 0:
                        del bytes_pair_freq[pair]
                    
                    bytes_pair_to_pretokens[pair].discard(affected_pretoken)
                    if len(bytes_pair_to_pretokens[pair]) == 0:
                        del bytes_pair_to_pretokens[pair]

                for i in range(len(new_pretoken) - 1):
                    pair = new_pretoken[i:i+2]
                    bytes_pair_freq[pair] += temp_freq    
                    bytes_pair_to_pretokens[pair].add(new_pretoken)
        
        return vocab, merges


if __name__ == "__main__":
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 500, special_tokens=["<|endoftext|>"])




