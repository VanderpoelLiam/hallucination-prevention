def read_lines(filename, start):
    lines = []
    order = []
    with open(filename) as file:
        for line in file:
            if start in line:
                l = list(map(str, line.rstrip().split()))
                clean_line = l[1:]
                clean_order = int(l[0].split(start)[1])
                lines.append(clean_line)
                order.append(clean_order)
    lines = [x for _, x in sorted(zip(order, lines))]
    return lines

def write_lines(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            f.write("%s\n" % " ".join(d))

if __name__ == '__main__':
    filename = "logs/experiments/generate_hallucination.log"
    ents = read_lines(filename, "ENT-")
    lang_ents = read_lines(filename, "ENT_LANG-")
    sm = read_lines(filename, "P_SM-")
    lm = read_lines(filename, "P_LM-")

    out_path = "data/xsum-hallucination/"
    write_lines(out_path + "test.entropy", ents)
    write_lines(out_path + "test.entropy.lang_full", lang_ents)
    write_lines(out_path + "test.prob.sm", sm)
    write_lines(out_path + "test.prob.lm", lm)
