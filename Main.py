#!/usr/bin/env python3

# Import dependencies
import os
import pandas as pd
import numpy as np
import argparse
import glob
from pathlib import Path
import igraph as ig
import matplotlib.pyplot as plt
from kneebow.rotor import Rotor
#from kneed import DataGenerator, KneeLocator

# Initiate parsing
def network_pipeline(input_file, outdir, tag):
    print("Running network pipeline with options:", input_file, outdir, tag)

    ## Handled arguments used
    # Sourmash input csv
    if input_file:
        if os.path.isfile(input_file):
            INPUT_FILE = input_file
        else:
            print("Input csv file does not exist! Exiting...")
            exit()
    else:
        print("No specified input file! Exiting...")
        exit()
    # Output directory
    if outdir:
        OUTDIR = Path(outdir)
        if OUTDIR.is_dir():
            print(f"Output directory {OUTDIR} exists.")
        else:
            print(f"Output directory {OUTDIR} does not exist. Creating it...")
            OUTDIR.mkdir(parents=True, exist_ok=True)
    else:
        print("No output directory specified, using default...")
        OUTDIR = Path("netpass_output")
        OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to {OUTDIR}")
    # Sequence tag
    if tag:
        TAG = tag
    else:
        print("No tag specified! Please provided a tag to describe sequence origins. Exiting...")
        exit()

    ### Convert matrix to edgetable
    ## Load in matrix
    wide_df = pd.read_csv(INPUT_FILE)
    ## Convert wide table to long format
    # Add genome names to first column
    genome_names = wide_df.columns.tolist()
    wide_df.insert(0, 'Source', genome_names)
    # Convert to long format
    wide_df_melted = wide_df.melt(id_vars=['Source'], var_name='Target', value_name='Value')
    # Remove self-loops
    wide_df_melted_noloops = wide_df_melted[wide_df_melted['Source'] != wide_df_melted['Target']]
    ## Remove duplicate edges
    # Sort the df to get duplicate edges in different orientations beside each other
    wide_df_sorted = wide_df_melted_noloops.copy()
    wide_df_sorted[['Source', 'Target']] = pd.DataFrame(np.sort(wide_df_melted_noloops[['Source', 'Target']], axis=1), index=wide_df_melted_noloops.index)
    # Drop duplicates
    wide_df_unique = wide_df_sorted.drop_duplicates(subset=['Source', 'Target'])
    # Save this file to the output directory
    output_path = OUTDIR / "Edgetable_NoDupesOrLoops.csv"
    wide_df_unique.to_csv(output_path, index=False)

    ### Find optimal edgetable to use
    ## Create Edgetable variations
    # Create a directory to hold these
    variations_outdir = OUTDIR / "Edgetable_Variations"
    variations_outdir.mkdir(parents=True, exist_ok=True)
    print("Creating edgetabele variations")
    # Initiate loop for removing edge weights
    for weight in range(0,100,1):
        print(f"Edgetable minimum weight: {weight}")
        # Keep only edges above the weight variation
        edgetable_variant = wide_df_unique[wide_df_unique['Value'] > (weight / 100)]
        # Set outfile name
        path = Path(INPUT_FILE)
        basename_noExt = path.stem
        outfile = f"{basename_noExt}_{weight}.csv"
        outfile_path = variations_outdir / outfile
        # Save edgetable variation
        edgetable_variant.to_csv(outfile_path, index=False)

    ## Get number of communities for each variation
    # Initialise lists for making the df of community count
    MinWeight = []
    NumComs = []
    print("Finding cumulative area under the curve")
    for weight in range(0,100,1):
        pattern = os.path.join(variations_outdir, f"*_{weight}.csv")
        tables = glob.glob(pattern)
        table = tables[0]
        df = pd.read_csv(table)
        # Make a graph using the edgetable
        g = ig.Graph.TupleList(df.itertuples(index=False), directed=False, weights=True)
        # Get the number of communities
        communities = g.community_multilevel(weights=g.es["weight"], return_levels=False)
        num_communities = len(communities)
        # Append data to the lists
        MinWeight.append(weight)
        NumComs.append(num_communities)
        print(f"Minimum Weight: {MinWeight[-1]}, Number Communities: {NumComs[-1]}")
    # Create a dataframe using the lists
    comsVweight = pd.DataFrame({'Minimum_Edge_Weight': MinWeight, 'Number_Communities': NumComs})
    # Save it to the output dir
    outfile_path = OUTDIR / "MinWeightVNumComs.csv"
    comsVweight.to_csv(outfile_path, index=False)
    print(comsVweight)
    # Create a scatter plot to show data
    plt.figure(figsize=(8,6))
    plt.scatter(comsVweight['Minimum_Edge_Weight'], comsVweight['Number_Communities'], color='blue', alpha=0.5)
    plt.title('Number of Communities vs. Minimum Edge Weight')
    plt.xlabel('Minimum Edge Weight')
    plt.ylabel('Number of Communities')
    plt.grid(True)
    plt.savefig('minweightVnumcoms.png', dpi=300, bbox_inches='tight')
    
    ## Get the AUC for the comsVweight graph
    # Initialise variables
    cumulative_area = 0
    prev_x = None
    prev_y = None
    AUC = pd.DataFrame(columns=['Minimum', 'CumulativeAUC'])
    # Loop through comsVweight and get cumulative area
    for index, row in comsVweight.iterrows():
        x, y = row['Minimum_Edge_Weight'], row['Number_Communities']
        # If statement to account for no variable atm
        if prev_x is not None:
            prev_y = float(prev_y)
            prev_x = float(prev_x)
            y = float(y)
            x = float(x)
            # Trapezium rule
            area = 0.5 * (y + prev_y) * (x - prev_x)
            cumulative_area += area
        # Set x,y as previous for next loop
        prev_x = x
        prev_y = y
        AUC = AUC._append({'Minimum': x, 'CumulativeAUC': cumulative_area}, ignore_index=True)
        # Append to lists
    print(AUC)

    ## Use the 'Kneedle' point to find the knee of the curve
    x = AUC['Minimum'].values
    x = [float(i) for i in x]
    y = AUC['CumulativeAUC'].values
    y = [float(i) for i in y]

    rotor = Rotor()
    rotor.fit_rotate(AUC)
    elbow_idx = rotor.get_elbow_index()
    print(f"Elbow of AUC curve is: {elbow_idx}")
    rotor.plot_elbow()
    # Save the plot as a PNG file
    plt.savefig("KneePointForMinEdgeWeight.png", dpi=300, bbox_inches='tight', format="png")
    # Optionally, you can close the plot if you are not displaying it
    plt.close()
    
    # Broken kneedle code
    #kneedle = KneeLocator(x,y, S=1.0, curve="concave", direction="increasing")
    #print(f"Knee point of curve (min. edge weight) is: {(round(kneedle.knee, 3))}")
    #kneedle.plot_knee()
    # Annotate the knee point
    #if knee_x is not None and knee_y is not None:
    #    plt.annotate(f'Knee Point\n({round(knee_x, 3)}, {round(knee_y, 3)})',
    #        xy=(knee_x, knee_y), xytext=(knee_x + 0.5, knee_y + 0.5),
    #        arrowprops=dict(facecolor='black', arrowstyle='->'))

def extract_pipeline(genomad, conservative):
    print("Running extract provirus pipeline with options:", genomad, conservative)

    ## Handle arguments used
    # Genomad input fastas
    if genomad:
        GEN_INPUT = Path(genomad)
        if os.path.isdir(GEN_INPUT):
            fna_files = glob.glob(os.path.join(GEN_INPUT, "*.fna"))
            if fna_files:
                GEN_FASTAS = fna_files
            else:
                print("No files with extension '.fna' detected in input directory! Exiting...")
                exit()
        else:
            print(f"{GEN_INPUT} does not exist! Exiting...")
            exit()
    else:
        print("No input directory containing geNomad provirus fastas specified! Exiting...")
        exit()
    # Set option for conservative topology
    if conservative:
        CONSERVATIVE = True
    else:
        CONSERVATIVE = False

def combine_pipeline(provirus, bacteria, procoms, baccoms):
    print("Running combine pipeline with options:", provirus, bacteria, procoms, baccoms)
    ## Handle argument used
    # Provirus edgetable
    if provirus:
        if os.path.isfile(provirus):
            PROV_EDGE = provirus
        else:
            print("Provirus edgetable not found! Exiting...")
            exit()
    else:
        print("No provirus edgetable specified! Exiting...")
        exit()
    # Bacteria edgetable
    if bacteria:
        if os.path.isfile(bacteria):
            BAC_EDGE = bacteria
        else:
            print("Bacteria edgetable not found! Exiting...")
            exit()
    else:
        print("No bacteria edgetable specified! Exiting...")
        exit()
    # Provirus community ID table
    if procoms:
        if os.path.isfile(procoms):
            PRO_COMS = procoms
        else:
            print("Provirus community ID table not found! Exiting...")
            exit()
    else:
        print("No provirus community ID table specified! Exiting...")
        exit()
    # Bacteria community ID table
    if baccoms:
        if os.path.isfile(baccoms):
            BAC_COMS = baccoms
        else:
            print("Bacteria community ID table not found! Exiting...")
            exit()
    else:
        print("No bacteria community ID table specified! Exiting...")
        exit()

def main():
    parser = argparse.ArgumentParser(description="Netpass - A community analysis of genome-genome networks")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    network_parser = subparsers.add_parser("network", help="Run network pipeline")
    network_parser.add_argument("-i", "--input", help="Sourmash compare output csv file")
    network_parser.add_argument("-o", "--outdir", help="Directory to output all results/files to")
    network_parser.add_argument("-t", "--tag", help="Identifier tag for sequences being used. Required. Reccomended: BAC/PRO for bacteria/provirus")
    

    extract_parser = subparsers.add_parser("extract", help="Run extract pipeline")
    extract_parser.add_argument("-g", "--genomad", help="Directory containing all geNomad provirus sequence fastas")
    extract_parser.add_argument("-c", "--conservative", help="Only keep sequences with topology of 'provirus'")


    combine_parser = subparsers.add_parser("combine", help="Run combine pipeline")
    combine_parser.add_argument("-p", "--provirus", help="Provirus edgetable from netpass network")
    combine_parser.add_argument("-b", "--bacteria", help="Bacteria edgetable from netpass network")
    combine_parser.add_argument("-P", "--procoms", help="Community membership table for proviruses from netpass network")
    combine_parser.add_argument("-B", "--baccoms", help="Community membership table for bacteria from netpass network")

    args = parser.parse_args()
    if args.subcommand == "network":
        network_pipeline(args.input, args.outdir, args.tag)
    elif args.subcommand == "extract":
        extract_pipeline(args.genomad, args.conservative)
    elif args.subcommand == "combine":
        combine_pipeline(args.provirus, args.bacteria, args.procoms, args.baccoms)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

