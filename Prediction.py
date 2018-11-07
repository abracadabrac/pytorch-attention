import argparse
import os
import csv

from data.vars import Vars

V = Vars()

"""
This script permits display the /Test/predictions.csv file
containing the prediction made by the net on the data-test

"""

if __name__ == "__main__":
    name_last_xp = os.listdir('%s/pytorch/' % V.experiments_folder)[-1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--name-xp', type=str, default=name_last_xp,
                        help='ex: 2018-09-04-18-05-26')
    args = parser.parse_args()

    with open('%s/pytorch/%s/Test/predictions.csv' % (V.experiments_folder, args.name_xp)) as f:
        reader = csv.DictReader(f)
        for ligne in reader:
            print(ligne['label'], ligne['prediction'], ligne['error'])
