#!/usr/bin/env python3

"""
Netpass - A community analysis tool for genome-genome networks

This tool analyzes genomic similarity networks to identify communities of related genomes
and automatically determines optimal network parameters for meaningful biological clustering.

Author: Emmet Campbell
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import glob
from pathlib import Path
import igraph as ig
import matplotlib.pyplot as plt
from kneebow.rotor import Rotor
import logging
from typing import Optional, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    """Main class for network analysis operations"""
    
    def __init__(self, input_file: str, outdir: str, tag: str, loops: str = "yes"):
        self.input_file = Path(input_file)
        self.outdir = Path(outdir)
        self.tag = tag
        self.keep_self_loops = loops.lower() == "yes"
        self.variations_outdir = self.outdir / "Edgetable_Variations"
        
        # Validate inputs
        self._validate_inputs()
        self._setup_directories()
    
    def _validate_inputs(self):
        """Validate input parameters"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file {self.input_file} does not exist")
        
        if not self.tag:
            raise ValueError("Tag must be provided to describe sequence origins")
            
        if not self.input_file.suffix.lower() == '.csv':
            logger.warning(f"Input file {self.input_file} doesn't have .csv extension")
    
    def _setup_directories(self):
        """Create output directories"""
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.variations_outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to {self.outdir}")
    
    def convert_matrix_to_edgetable(self) -> pd.DataFrame:
        """Convert similarity matrix to edgetable format"""
        logger.info("Converting similarity matrix to edgetable format")
        
        try:
            # Load similarity matrix
            wide_df = pd.read_csv(self.input_file, index_col=0)
            logger.info(f"Loaded matrix with {wide_df.shape[0]} genomes")
            
            # Convert to long format
            wide_df_melted = wide_df.reset_index().melt(
                id_vars=['index'], 
                var_name='Target', 
                value_name='Value'
            ).rename(columns={'index': 'Source'})
            
            # Handle self-loops
            if not self.keep_self_loops:
                wide_df_melted = wide_df_melted[wide_df_melted['Source'] != wide_df_melted['Target']]
                logger.info("Self-loops removed")
            
            # Remove duplicate edges (undirected network)
            wide_df_sorted = wide_df_melted.copy()
            wide_df_sorted[['Source', 'Target']] = pd.DataFrame(
                np.sort(wide_df_melted[['Source', 'Target']], axis=1), 
                index=wide_df_melted.index
            )
            
            # Drop duplicates and NaN values
            wide_df_unique = wide_df_sorted.drop_duplicates(subset=['Source', 'Target'])
            wide_df_unique = wide_df_unique.dropna()
            
            # Save processed edgetable
            suffix = "NoDupesOrLoops" if not self.keep_self_loops else "NoDupes"
            output_path = self.outdir / f"Edgetable_{suffix}.csv"
            wide_df_unique.to_csv(output_path, index=False)
            logger.info(f"Processed edgetable saved to {output_path}")
            
            return wide_df_unique
            
        except Exception as e:
            logger.error(f"Error processing matrix: {e}")
            raise
    
    def create_edgetable_variations(self, edgetable: pd.DataFrame, weight_step: int = 1) -> None:
        """Create edgetable variations with different minimum weights"""
        logger.info("Creating edgetable variations")
        
        basename = self.input_file.stem
        
        for weight in range(0, 100, weight_step):
            # Filter edges above threshold
            edgetable_variant = edgetable[edgetable['Value'] > (weight / 100)]
            
            # Skip if no edges remain
            if len(edgetable_variant) == 0:
                logger.warning(f"No edges remain at weight threshold {weight}")
                continue
            
            # Save variation
            outfile_path = self.variations_outdir / f"{basename}_{weight}.csv"
            edgetable_variant.to_csv(outfile_path, index=False)
        
        logger.info(f"Created {len(list(self.variations_outdir.glob('*.csv')))} edgetable variations")
    
    def analyze_community_structure(self, weight_step: int = 1) -> pd.DataFrame:
        """Analyze community structure across different weight thresholds"""
        logger.info("Analyzing community structure across weight thresholds")
        
        min_weights = []
        num_communities = []
        basename = self.input_file.stem
        
        for weight in range(0, 100, weight_step):
            file_path = self.variations_outdir / f"{basename}_{weight}.csv"
            
            if not file_path.exists():
                logger.warning(f"File {file_path} not found, skipping weight {weight}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Skip if empty
                if len(df) == 0:
                    logger.warning(f"Empty edgetable at weight {weight}")
                    continue
                
                # Create network
                g = ig.Graph.TupleList(df.itertuples(index=False), directed=False, weights=True)
                
                # Find communities
                communities = g.community_multilevel(weights=g.es["weight"], return_levels=False)
                num_coms = len(communities)
                
                min_weights.append(weight)
                num_communities.append(num_coms)
                
                logger.debug(f"Weight {weight}: {num_coms} communities")
                
            except Exception as e:
                logger.error(f"Error analyzing weight {weight}: {e}")
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Minimum_Edge_Weight': min_weights, 
            'Number_Communities': num_communities
        })
        
        # Save results
        output_path = self.outdir / "MinWeightVNumComs.csv"
        results_df.to_csv(output_path, index=False)
        
        # Create visualization
        self._plot_communities_vs_weight(results_df)
        
        logger.info(f"Community analysis complete. Results saved to {output_path}")
        return results_df
    
    def _plot_communities_vs_weight(self, data: pd.DataFrame) -> None:
        """Create scatter plot of communities vs weight"""
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Minimum_Edge_Weight'], data['Number_Communities'], 
                   color='blue', alpha=0.7, s=50)
        plt.title('Number of Communities vs. Minimum Edge Weight', fontsize=14)
        plt.xlabel('Minimum Edge Weight (%)', fontsize=12)
        plt.ylabel('Number of Communities', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.outdir / f'communities_vs_weight_{self.tag}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Community plot saved to {plot_path}")
    
    def find_optimal_threshold(self, communities_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
        """Find optimal threshold using area under curve and elbow detection"""
        logger.info("Finding optimal threshold using elbow detection")
        
        # Calculate cumulative area under curve
        auc_data = self._calculate_auc(communities_df)
        
        # Find elbow point
        try:
            rotor = Rotor()
            rotor.fit_rotate(auc_data)
            elbow_idx = rotor.get_elbow_index()
            optimal_weight = int(auc_data.iloc[elbow_idx]['Minimum'])
            
            logger.info(f"Optimal threshold found: {optimal_weight}% similarity")
            
            # Create and save elbow plot
            self._plot_elbow_detection(rotor, optimal_weight)
            
            # Save AUC data
            auc_output_path = self.outdir / f'CumulativeAUC_{optimal_weight}.csv'
            auc_data.to_csv(auc_output_path, index=False)
            
            return optimal_weight, auc_data
            
        except Exception as e:
            logger.error(f"Error in elbow detection: {e}")
            # Fallback to simple heuristic
            logger.info("Using fallback heuristic for threshold selection")
            mid_point = len(communities_df) // 2
            optimal_weight = int(communities_df.iloc[mid_point]['Minimum_Edge_Weight'])
            return optimal_weight, auc_data
    
    def _calculate_auc(self, communities_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate area under curve for community data"""
        cumulative_area = 0
        auc_data = []
        
        for i in range(len(communities_df)):
            if i > 0:
                # Trapezoidal rule
                x_curr = communities_df.iloc[i]['Minimum_Edge_Weight']
                x_prev = communities_df.iloc[i-1]['Minimum_Edge_Weight']
                y_curr = communities_df.iloc[i]['Number_Communities']
                y_prev = communities_df.iloc[i-1]['Number_Communities']
                
                area = 0.5 * (y_curr + y_prev) * (x_curr - x_prev)
                cumulative_area += area
            
            auc_data.append({
                'Minimum': communities_df.iloc[i]['Minimum_Edge_Weight'],
                'CumulativeAUC': cumulative_area
            })
        
        return pd.DataFrame(auc_data)
    
    def _plot_elbow_detection(self, rotor: Rotor, optimal_weight: int) -> None:
        """Create elbow detection plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        rotor.plot_elbow(ax=ax)
        ax.set_title(f'Elbow Detection - Optimal Weight: {optimal_weight}%', fontsize=14)
        ax.set_xlabel('Minimum Edge Weight (%)', fontsize=12)
        ax.set_ylabel('Cumulative AUC', fontsize=12)
        
        plot_path = self.outdir / f"elbow_detection_{optimal_weight}_{self.tag}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Elbow detection plot saved to {plot_path}")
    
    def generate_community_mapping(self, optimal_weight: int) -> pd.DataFrame:
        """Generate genome to community mapping using optimal threshold"""
        logger.info(f"Generating community mapping with threshold {optimal_weight}%")
        
        # Load optimal edgetable
        basename = self.input_file.stem
        edgetable_path = self.variations_outdir / f"{basename}_{optimal_weight}.csv"
        
        if not edgetable_path.exists():
            raise FileNotFoundError(f"Optimal edgetable not found: {edgetable_path}")
        
        edgetable = pd.read_csv(edgetable_path)
        
        # Create network and find communities
        g = ig.Graph.TupleList(edgetable.itertuples(index=False), directed=False, weights=True)
        communities = g.community_multilevel(weights=g.es["weight"], return_levels=False)
        
        logger.info(f"Found {len(communities)} communities with {g.vcount()} genomes")
        
        # Create genome to community mapping
        genome_to_community = {}
        for community_id, community_nodes in enumerate(communities):
            for node in community_nodes:
                genome_id = g.vs[node]["name"]
                genome_to_community[genome_id] = community_id
        
        # Create dataframe
        mapping_df = pd.DataFrame(
            list(genome_to_community.items()), 
            columns=["Genome_ID", "Community"]
        )
        
        # Save mapping
        mapping_path = self.outdir / f"genome_community_mapping_{optimal_weight}_{self.tag}.csv"
        mapping_df.to_csv(mapping_path, index=False)
        
        # Create intra-community edgetable
        self._create_intra_community_edgetable(edgetable, mapping_df, optimal_weight)
        
        logger.info(f"Community mapping saved to {mapping_path}")
        return mapping_df
    
    def _create_intra_community_edgetable(self, edgetable: pd.DataFrame, 
                                        mapping_df: pd.DataFrame, optimal_weight: int) -> None:
        """Create edgetable with only intra-community edges"""
        # Merge with community mappings
        merged_df = edgetable.merge(
            mapping_df, left_on='Source', right_on='Genome_ID', how='left'
        ).merge(
            mapping_df, left_on='Target', right_on='Genome_ID', 
            how='left', suffixes=('_source', '_target')
        )
        
        # Filter for same-community edges
        intra_community = merged_df[
            merged_df['Community_source'] == merged_df['Community_target']
        ][['Source', 'Target', 'Value']]
        
        # Save intra-community edgetable
        basename = self.input_file.stem
        output_path = self.outdir / f"{basename}_{optimal_weight}_intra_community_{self.tag}.csv"
        intra_community.to_csv(output_path, index=False)
        
        logger.info(f"Intra-community edgetable saved to {output_path}")

def run_network_pipeline(input_file: str, outdir: str, tag: str, loops: str = "yes", 
                        weight_step: int = 1) -> None:
    """Main pipeline function"""
    logger.info(f"Starting network analysis pipeline")
    logger.info(f"Input: {input_file}, Output: {outdir}, Tag: {tag}, Loops: {loops}")
    
    try:
        # Initialize analyzer
        analyzer = NetworkAnalyzer(input_file, outdir, tag, loops)
        
        # Convert matrix to edgetable
        edgetable = analyzer.convert_matrix_to_edgetable()
        
        # Create variations
        analyzer.create_edgetable_variations(edgetable, weight_step)
        
        # Analyze community structure
        communities_df = analyzer.analyze_community_structure(weight_step)
        
        if len(communities_df) == 0:
            logger.error("No valid community data generated")
            return
        
        # Find optimal threshold
        optimal_weight, auc_data = analyzer.find_optimal_threshold(communities_df)
        
        # Generate community mapping
        mapping_df = analyzer.generate_community_mapping(optimal_weight)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Optimal similarity threshold: {optimal_weight}%")
        logger.info(f"Number of communities: {len(mapping_df['Community'].unique())}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Netpass - A community analysis tool for genome-genome networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s network -i similarity_matrix.csv -o results/ -t BAC
  %(prog)s network -i matrix.csv -o output/ -t PRO -l no --weight-step 5
        """
    )
    
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand", help="Available commands")
    
    # Network pipeline
    network_parser = subparsers.add_parser("network", help="Run network analysis pipeline")
    network_parser.add_argument("-i", "--input", required=True,
                               help="Input similarity matrix CSV file (from sourmash compare)")
    network_parser.add_argument("-o", "--outdir", required=True,
                               help="Output directory for results")
    network_parser.add_argument("-t", "--tag", required=True,
                               help="Identifier tag for sequences (e.g., BAC, PRO)")
    network_parser.add_argument("-l", "--loops", default="yes", choices=["yes", "no"],
                               help="Include self-loops in network (default: yes)")
    network_parser.add_argument("--weight-step", type=int, default=1, 
                               help="Step size for weight thresholds (default: 1)")
    network_parser.add_argument("-v", "--verbose", action="store_true",
                               help="Enable verbose logging")
    
    # Extract pipeline (placeholder)
    extract_parser = subparsers.add_parser("extract", help="Extract provirus sequences (not implemented)")
    extract_parser.add_argument("-g", "--genomad", help="Directory containing geNomad provirus fastas")
    extract_parser.add_argument("-c", "--conservative", action="store_true",
                               help="Use conservative topology filtering")
    
    # Combine pipeline (placeholder)
    combine_parser = subparsers.add_parser("combine", help="Combine analysis results (not implemented)")
    combine_parser.add_argument("-p", "--provirus", help="Provirus edgetable")
    combine_parser.add_argument("-b", "--bacteria", help="Bacteria edgetable")
    combine_parser.add_argument("-P", "--procoms", help="Provirus community mapping")
    combine_parser.add_argument("-B", "--baccoms", help="Bacteria community mapping")
    
    args = parser.parse_args()
    
    # Set logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run appropriate pipeline
    if args.subcommand == "network":
        run_network_pipeline(args.input, args.outdir, args.tag, args.loops, args.weight_step)
    elif args.subcommand == "extract":
        logger.error("Extract pipeline not yet implemented")
        sys.exit(1)
    elif args.subcommand == "combine":
        logger.error("Combine pipeline not yet implemented")
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
