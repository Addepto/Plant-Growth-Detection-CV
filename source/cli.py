import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='CLI for the KWS project.')
    parser.add_argument_group('Input definition', 'Info needed to get correct images for the analysis.')
    parser.add_argument('--input_dir', type=str, help='Path to the directory with images from the client.',
                        required=True)
    parser.add_argument('--plants_names', type=str, nargs='+', default=['Beta vulgaris'],
                        help='Which plants should be analyzed (names of the directories).', )
    parser.add_argument('--growth_stages', type=str, nargs='+', default=['Cotyledon'],
                        help='Which growth stages should be analyzed (names of the directories)')
    return parser
