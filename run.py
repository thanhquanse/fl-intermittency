import os
import sys
import argparse

def main(argv):
    IPYNB_FILENAME = 'FLDigitalTwin.ipynb'
    CONFIG_FILENAME = '.config_ipynb'
    OUTPUT_DIR = "outputs"

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--dataset", type=str, default="traffic")
    _parser.add_argument("--prefix", type=str, default="normal")
    _parser.add_argument("--percent_mc", type=str, default="01")
    _parser.add_argument("--missing_mode", type=str, default="noajacency")
    _parser.add_argument("--matrix_ml", type=str, default="10x10")
    _parser.add_argument("--is_cluster", type=str, default="no")
    _parser.add_argument("--weight_mechanism", type=int, default=0)
    
    parser = _parser.parse_args(argv[1:])

    with open(CONFIG_FILENAME,'w') as f:
        f.write(' '.join(argv))
    
    if parser.is_cluster != 'no':
        OUTPUT_DIR = "outputs/" + parser.is_cluster

    if parser.prefix == 'normal':
        output_filename = f"Results-{parser.dataset}-{parser.prefix}-{parser.matrix_ml}"
    else:
        output_filename = f"Results-{parser.dataset}-{parser.prefix}-{parser.matrix_ml}-{parser.missing_mode}-{parser.percent_mc}"
    os.system('jupyter nbconvert --execute --to notebook --output {:s} --output-dir {:s} {:s}'.format(output_filename, OUTPUT_DIR, IPYNB_FILENAME))
    return None

if __name__ == "__main__":
    main(sys.argv)