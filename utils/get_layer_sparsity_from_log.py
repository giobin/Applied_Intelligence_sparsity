import re
import argparse

regex = '([-+]?\d*\.?\d+) out of ([-+]?\d*\.?\d+)'

def get_number(input_regex, input):
    match = re.search(input_regex, input, re.IGNORECASE)
    if match:
        number_1 = f'{match[1]}'
        number_2 = f'{match[2]}'
        return float(number_1), float(number_2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', type=str)
    args = parser.parse_args()
    print(f"gathering from {args.log}")

    with open(args.log, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 94

    emb_lines = lines[:2] + lines[34:37] + lines[89:92]
    enc_att = lines[2:12] + lines[18:28]
    enc_fnn = lines[12:18] + lines[28:34]
    dec_att = lines[37:57] + lines[63:83]
    dec_fnn = lines[57:63] + lines[83:89]

    names = ["enc_att","dec_att","enc_fnn","dec_fnn","emb_lines"]
    for ix, l in enumerate([enc_att, dec_att, enc_fnn, dec_fnn, emb_lines]):
        remained_counter = 0
        out_of_counter = 0
        for e in l:
            remained, out_of = get_number(regex, e)
            print(f"remained: {remained}, out_of: {out_of}")
            remained_counter += remained
            out_of_counter += out_of
        print(f"{names[ix]} remained/out_of: {remained_counter/out_of_counter:.4f}\n")




