
""" 
Level 3 filtering
- remove tweets with mentions
- remote tweets with images (start with a https://t.co link)
"""

import sys

input_file = sys.argv[1]
outfile = sys.argv[2]

with open(f"{input_file}.txt") as tweets:
    texts = tweets.read().splitlines()
    # print(texts)
    texts_ = []
    for text in range(0, len(texts)-1):
        # print(texts[text])
        if "@" in texts[text] or "https" in texts[text] or "#" in texts[text]:
            pass
        else:
            texts_.append(texts[text])
        # print(texts_)

with open(f"{outfile}.txt", "w") as f:
    for i in range(0, len(texts_)-1):
        f.write(f"{texts_[i]}\n")
