import tqdm
from  tokenizers.normalizers import *
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

files = [f"data/tokenizer_data.txt"]

def data_iterator():
    # Removes all characters, that are only used once in the entire dataset
    chars_used_once = set()
    char_counter = {}

    for file in files:
        with open(file, "r") as f:
            for line in tqdm.tqdm(f.readlines(), desc=f"Counting characters {file}"):
                for char in line:
                    if char in char_counter:
                        char_counter[char] += 1
                    else:
                        char_counter[char] = 1

    for char, count in char_counter.items():
        if count <= 2:
            chars_used_once.add(char)

    print(f"Removing characters:")
    print(chars_used_once)

    for file in files:
        with open(file, "r") as f:
            for line in tqdm.tqdm(f.readlines(), desc=f"Processing {file}"):
                line_without_chars = "".join([char for char in line if char not in chars_used_once])
                yield line_without_chars

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size = 255, show_progress=True, end_of_word_suffix="</w>")
tokenizer.pre_tokenizer = pre_tokenizers.Sequence( [pre_tokenizers.WhitespaceSplit() ])
tokenizer.normalizer = Sequence([Lowercase(), NFD(), StripAccents()])
tokenizer.train_from_iterator(data_iterator(), trainer=trainer)
tokenizer.decoder = decoders.BPEDecoder()

tokenizer.save("tokenizer.json")

encoded = tokenizer.encode("This is a test!")
print(encoded.ids)

decoded = tokenizer.decode(encoded.ids)
print(decoded)

# Try to load the tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
encoded = tokenizer.encode("This is á test!")
print(encoded.ids)
decoded = tokenizer.decode(encoded.ids)
print(decoded)
