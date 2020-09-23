# imports
import operator
import os
import csv
import pyexcel
import pickle
import yaml
import pprint
import json
import re

import numpy as np
import itertools as itt
from pycompomics.pycompomics import SearchGUI, PeptideShaker
from pyslurm_computecanada.pyslurm_computecanada.pyslurm import Slurm

from Bio import SeqIO
import subprocess
import pyfaidx
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
import pandas as pd
from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
import scipy.stats as stats

from sklearn.metrics import precision_recall_curve, roc_curve
from PS_report_parser.hierarchical_report_parser import *

def venn(data, file_path=None, spacing=0.9):
    """ data is a list dict with labels as keys and sets as values
    """
    def get_venn(d, ax):
        labels, sets = zip(*list(d.items()))
        if len(d)==2:
            venn2(sets, set_labels=labels, ax=ax)
            venn2_circles(sets, ax=ax)
        elif len(d)==3:
            venn3(sets, set_labels=labels, ax=ax)
            venn3_circles(sets, ax=ax)
        else:
            return 'Error, data dict must contain 2 or 3 sets'
        
    n_axes = len(data)
    if n_axes>1:
        figsize = (8, 8*n_axes)
    else:
        figsize = (3,3)
    fig, axs = plt.subplots(1, n_axes, figsize=figsize)
    if n_axes == 1:
        get_venn(data[0], axs)
    else:
        for n in range(n_axes):
            get_venn(data[n], axs[n])
        
    if file_path is not None:
        plt.savefig(file_path)
    plt.subplots_adjust(wspace=spacing)
    plt.show()

def is_alt(acc):
    if any(x in acc for x in {'IP_', 'II_'}):
        return True
    return False

fields = ['OS', 'GN', 'TA', 'PA']
def parse_fasta_header(h):
    h = h.split()
    acc = h[0].split('|')[0]
    res = {}
    for f in h[1:]:
        for field in fields:
            if f[:2] == field:
                res[field] = f[3:]
    if 'GN' not in res:
        res['GN'] = 'unknown'
    if 'PA' in res:
        res['PA'] = res['PA'].split(',')
    return res 

prot_gene_dict = {}
for record in SeqIO.parse('human-openprot-r0_0-refprots+altprots+isoforms-+uniprot2019_03_01.fasta', 'fasta'):
    header = parse_fasta_header(record.description)
    if 'PA' not in header:
        prot_acc = record.name.split('|')[0]
        prot_gene_dict[prot_acc] = header['GN']
        continue
    for prot_acc in header['PA']:
        if prot_acc not in prot_gene_dict:
            prot_gene_dict[prot_acc] = header['GN']

def get_prey_gene(p_acc):
    p_acc_noversion = p_acc.split('.')[0]
    gene = p_acc
    if p_acc in prot_gene_dict and prot_gene_dict[p_acc]:
        gene = prot_gene_dict[p_acc].split()[0]
    elif p_acc_noversion in prot_gene_dict and prot_gene_dict[p_acc_noversion]:
        gene = prot_gene_dict[p_acc_noversion].split()[0]
    if ',' in gene:
        genes = gene.split(',')
        if not any('.' in g for g in genes):
            gene = genes[0]
        else:
            gene = [g for g in genes if '.' not in g][0]
    return gene

with open('uniprot_HGNC.tab', 'r') as f:
    for n,l in enumerate(f):
        ls = l.strip().split('\t')
        if n==0:
            keys = ls
            continue
        line = dict(zip(keys, ls))
        prot_gene_dict[line['Entry']] = line['Entry name'].split('_')[0]

ensp_genes = {}
with open('ensp_genes.tsv', 'r') as f:
    for n,l in enumerate(f):
        ls =l.strip().split('\t')
        if n==0:
            keys=ls
            continue
        if len(ls)<2: continue
        line = dict(zip(keys, ls))
        ensp_genes[line['Protein stable ID']] = line['Gene name']

uniprot_all_protaccs = set()
for seq in SeqIO.parse('swissprot_UP000005640.fasta', 'fasta'):
    prot_acc = seq.name.split('|')[1]
    uniprot_all_protaccs.add(prot_acc)

OP_all_refprotseq = set()
for n,fasta_entry in enumerate(SeqIO.parse('human-openprot-r0_0-refprots+altprots+isoforms-+uniprot2019_03_01.fasta', 'fasta')):
    prot_acc = fasta_entry.name.split('|')[0]
    if not is_alt(prot_acc):
        OP_all_refprotseq.add(str(fasta_entry.seq))

def parse_gtf(fpath):
    gtf = []
    keys = ['chrom', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'desc']
    with open(fpath, 'r') as f:
        for n,l in enumerate(f):
            ls = l.strip().split('\t')
            line = dict(zip(keys, ls))
            line['desc'] = {x.split()[0]:x.split()[1].replace('"', '') for x in line['desc'][:-1].split(';')}
            for k in ['FPKM', 'frac', 'conf_lo', 'conf_hi', 'cov']:
                if k in line['desc']:
                    line['desc'][k] = float(line['desc'][k])
            gtf.append(line)
    return gtf

chroms = [str(x) for x in range(1,22)] + ['X', 'Y', 'MT']
aa_weights = { 
    'X': 110, 'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121, 'E': 147, 'Q': 146,
    'G': 75, 'H': 155, 'I': 131, 'L':131,'K': 146, 'M': 149, 'F': 165, 'P': 115,
    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'U':168, 'V': 117
}
gencode = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'
    }

def translate(rna, frame=0):
    rna = rna[frame:]
    codons = [rna[n:n+3] for n in range(0,len(rna),3)]
    if not codons:
        return ''
    if len(codons[-1])<3:
        codons = codons[:-1]
    translation = ''
    for codon in codons:
        if any(x in codon for x in 'NH') or codon not in gencode:
            translation += 'X'
            continue
        translation += gencode[codon]
    return translation

def altorfs_3frame(seq):
    orfs = []
    for f in range(4):
        seq = translate(str(seq), frame=f)
        for orf in seq.split('_'):
            if 'M' not in orf: continue
            M_index = orf.find('M')

            orf = orf[M_index:]
            if len(orf)>=30 and 'X' not in orf:
                orfs.append(orf)
    return orfs

def truncate(seq, length=80):
    return '\n'.join([seq[n:n+length] for n in range(0, len(seq), length)])

def is_altpep_unique(pep_seq):
    for ref_seq in list(OP_all_refprotseq):
        if pep_seq in ref_seq:
            return False
    return True

def get_protgrp_psms(fpath):
    report = parse(fpath)
    prot_grps = {}
    for prot_grp_n, protein_grp in enumerate(report):
        for peptide in protein_grp.peptide_rows:
            prot_grp_id = protein_grp.row[0][0] # re-define for every peptide
            pep_seq = peptide.row[2]
            prot_accs_genes = []
            for p in peptide.peptide_names:
                prot_acc = p.strip().split('|')[0]
                if prot_acc not in prot_gene_dict and 'CUFF' not in prot_acc:
                    if prot_acc not in op75_to_78 and prot_acc not in zikv_baits and 'ZIKV' not in prot_acc:
                        continue # lost at OP 1.6
                    elif prot_acc not in zikv_baits and 'ZIKV' not in prot_acc:
                        prot_acc = op75_to_78[prot_acc] # fix changes in tt_type, release 1.6 (Aug 25 2020)
                if 'CUFF' in prot_acc:
                    prot_gene = 'CUFF_gene'
                    if not is_altpep_unique(pep_seq): continue
                elif prot_acc in zikv_baits or 'ZIKV' in prot_acc:
                    prot_gene = 'ZIKA'
                else:
                    prot_gene = prot_gene_dict[prot_acc] # *** check that uniprot gene mapping are consistent with Shah
                prot_accs_genes.append((prot_acc, prot_gene))
            
            tt_type_isalt = [is_alt(acc) for acc, gene in prot_accs_genes]
            if any(tt_type_isalt) and not all(tt_type_isalt):                                         # if any but not all members of pep_grp are alt,
                prot_accs_genes = list(itt.compress(prot_accs_genes, [not x for x in tt_type_isalt])) # remove them 
            
            protgrp_isalt = False
            if all(tt_type_isalt):
                protgrp_isalt = True
                if 'alt' not in prot_grp_id:
                    prot_grp_id = prot_grp_id + '_alt'

            for psm in peptide.psm_rows:
                if 'Confident' not in psm.validation: continue
                if protgrp_isalt:
                    if not is_altpep_unique(pep_seq): continue
                    if prot_grp_id not in prot_grps:
                        prot_grps[prot_grp_id] = {
                                                  'prot_accs_genes':set(prot_accs_genes), 
                                                  'psms':[(pep_seq, psm.spectrum_title)],
                                                  'prot_accs_genes_psms_cnt': dict(zip(prot_accs_genes, [1,]*len(prot_accs_genes)))
                                                 }
                    else:
                        prot_grps[prot_grp_id]['prot_accs_genes'] = prot_grps[prot_grp_id]['prot_accs_genes'].union(set(prot_accs_genes))
                        for pag in prot_accs_genes:
                            if pag not in prot_grps[prot_grp_id]['prot_accs_genes_psms_cnt']:
                                prot_grps[prot_grp_id]['prot_accs_genes_psms_cnt'][pag] = 0
                            prot_grps[prot_grp_id]['prot_accs_genes_psms_cnt'][pag] += 1
                        prot_grps[prot_grp_id]['psms'].append((pep_seq, psm.spectrum_title))
                else:
                    
                    if prot_grp_id not in prot_grps:
                        prot_grps[prot_grp_id] = {
                                                  'prot_accs_genes':set(prot_accs_genes), 
                                                  'psms':[(pep_seq, psm.spectrum_title)],
                                                  'prot_accs_genes_psms_cnt': dict(zip(prot_accs_genes, [1,]*len(prot_accs_genes)))
                                                 }
                    else:
                        prot_grps[prot_grp_id]['prot_accs_genes'] = prot_grps[prot_grp_id]['prot_accs_genes'].union(set(prot_accs_genes))
                        for pag in prot_accs_genes:
                            if pag not in prot_grps[prot_grp_id]['prot_accs_genes_psms_cnt']:
                                prot_grps[prot_grp_id]['prot_accs_genes_psms_cnt'][pag] = 0
                            prot_grps[prot_grp_id]['prot_accs_genes_psms_cnt'][pag] += 1
                        prot_grps[prot_grp_id]['psms'].append((pep_seq, psm.spectrum_title))
    return prot_grps

def remove_invalid_grp(protgrp_psms, valid_prot_accs):
    item_ls = protgrp_psms.copy().items()
    for grp_id, grp in item_ls:
        if len(grp['prot_accs_genes'].intersection(valid_prot_accs))==0:
            del protgrp_psms[grp_id]
    return protgrp_psms

def get_single_gene_psms(protgrp_psms):
    single_gene_psms = {}
    for grp_id, grp in protgrp_psms.items():
        prot_accs, genes = zip(*grp['prot_accs_genes'])
        
        if all(is_alt(x) for x in prot_accs): # extra check
            gene_alt_acc, psm_cnt = sorted(grp['prot_accs_genes_psms_cnt'].items(), key=lambda x: [-x[1], x[0]])[0]
            gene = '|'.join(gene_alt_acc)
            single_gene_psms[grp_id] = {'gene':gene, 'psms':grp['psms']}
        elif len(set(genes)) == 1:
            gene = genes[0]
            single_gene_psms[grp_id] = {'gene':gene, 'psms':grp['psms']}
            
    # group single gene prot groups under gene
    psms = {}
    single_genes = set()
    for gene, psms_grp in itt.groupby(sorted(single_gene_psms.values(), key=lambda x: x['gene']), key=lambda x: x['gene']):
        psms_ls = [x for y in [x['psms'] for x in psms_grp] for x in y]
        n_upep = len(set(x[0] for x in psms_ls))
        if n_upep>1 or is_alt(gene):   # check for minimum of 2 unique peptide if not alt
            psms[gene] = {'psms':psms_ls, 'n_upep':n_upep}
            single_genes.add(gene)
    return psms, single_genes, set(single_gene_psms.keys())

def get_multigene_psms(protgrp_psms, bait):
    rep_psms, singles_genes, singles_grpids = get_single_gene_psms(protgrp_psms)
    for grp_id, grp in protgrp_psms.items():
        if grp_id in singles_grpids: continue
        prot_accs, genes = zip(*grp['prot_accs_genes'])
        already_detected = set(genes).intersection(singles_genes)
        if len(already_detected)>1:
            for gene in list(already_detected):
                rep_psms[gene]['psms'].extend(grp['psms'])
    return rep_psms

def clean_all_grps(protgrp_psms, valid_prot_accs):
    for grp_id, prot_grp in protgrp_psms.items():
        prot_accs_genes = prot_grp['prot_accs_genes'].intersection(valid_prot_accs)
        if len(prot_accs_genes)>0:
            protgrp_psms[grp_id]['prot_accs_genes'] = prot_accs_genes
        else: # prot group is not valid
            del protgrp_psms[grp_id]
    return protgrp_psms

def get_all_psms(bait, condition):
    psms = {
        bait:{
            'exps':{exp_id:{'protgrp_psms':get_protgrp_psms(exp_report_paths[condition][exp_id])} for exp_id in bait_rep_dict[bait] if exp_id in exp_report_paths[condition]},
            'valid_prot_accs':set(),
        }
    }
    for exp_id, exp in psms[bait]['exps'].items(): # Obtain a set of all proteins detected for each rep, take the intersection
        exp['all_prots'] = set(x for y in [list(x['prot_accs_genes']) for x in exp['protgrp_psms'].values()] for x in y)
    
    count_rep_per_prey = Counter([x for y in [x['all_prots'] for x in psms[bait]['exps'].values()] for x in y])
    valid_prot_accs = set(prey for prey, cnt in count_rep_per_prey.items() if cnt>1)
    psms[bait]['valid_prot_accs'] = valid_prot_accs
    
    for exp_id, exp in psms[bait]['exps'].items(): # Remove protein groups containing solely proteins seen in only 1 rep
        exp['protgrp_psms_valid'] = remove_invalid_grp(exp['protgrp_psms'], valid_prot_accs)

    for exp_id, exp in psms[bait]['exps'].items():
        exp['protgrp_psms_valid_clean'] = clean_all_grps(exp['protgrp_psms_valid'], valid_prot_accs)
            
    for exp_id, exp in psms[bait]['exps'].items(): # assign PSMs to prot groups, single gene and multigene
        exp['psms'] = get_multigene_psms(exp['protgrp_psms_valid_clean'], bait)
        
    return psms

def count_psms(apms_psms, crap, return_multi_prot_grp=False, ):
    apms_psms_cnt = {}
    multi_prot_grp = []
    n_protgrp_rep = 0
    for bait, exps in apms_psms.items():
        for exp_id, psms in exps['exps'].items():
            apms_psms_cnt[exp_id] = {}
            for prot_grp_id, prot_grp in psms['protgrp_psms_valid_clean'].items():
                n_protgrp_rep += 1
                if len(prot_grp['prot_accs_genes']) == 1: # only one prot in grp
                    prot_acc_gene = list(prot_grp['prot_accs_genes'])[0]
                else: # MORE than one prot in grp
                    if not any(p in shah_all_preys for p, g in prot_grp['prot_accs_genes']):
                        # take the one with most psms
                        prot_acc_gene = max(prot_grp['prot_accs_genes_psms_cnt'].items(), key=operator.itemgetter(1))[0]
                    else:
                        # take the one with most psms that is also in Shah
                        prot_grp['prot_accs_shared_shah'] = [pag for pag, c in prot_grp['prot_accs_genes_psms_cnt'].items() if pag[0] in shah_all_preys]
                        max_psm_cnt = max(prot_grp['prot_accs_genes_psms_cnt'].items(), key=operator.itemgetter(1))[1]
                        max_psm_prot_accs_genes = [pag for pag, c in prot_grp['prot_accs_genes_psms_cnt'].items() if c==max_psm_cnt]
                        if any(pag[0] in shah_all_preys for pag in max_psm_prot_accs_genes):
                            prot_acc_gene = [pag for pag in max_psm_prot_accs_genes if pag[0] in shah_all_preys][0]
                        else:
                            prot_acc_gene = max_psm_prot_accs_genes[0]

                        prot_grp['selected_prot_acc_gene'] = prot_acc_gene
                    prot_grp['selected_prot_acc_gene'] = prot_acc_gene
                    multi_prot_grp.append(prot_grp)

                if prot_acc_gene[0] in apms_psms_cnt[exp_id]: # sum psms if prot_acc_gene has already been selected for an other grp
                    apms_psms_cnt[exp_id][prot_acc_gene[0]] += prot_grp['prot_accs_genes_psms_cnt'][prot_acc_gene]
                if get_prey_gene(prot_acc_gene[0]) in crap: continue
                apms_psms_cnt[exp_id][prot_acc_gene[0]] = prot_grp['prot_accs_genes_psms_cnt'][prot_acc_gene]
                
    if return_multi_prot_grp:
        return apms_psms_cnt, multi_prot_grp
    return apms_psms_cnt


def parse_mist_metrics(fpath, weights):
    metrics = []
    interactions = set()
    with open(fpath, 'r') as f:
        for n,l in enumerate(f):
            ls = l.strip().split('\t')
            if n==0:
                keys = ls
                continue
                
            line = dict(zip(keys, ls))
            line['mist_score'] = 0.
            for k in ['Reproducibility', 'Abundance', 'Specificity']:
                line[k] = float(line[k])
                line['mist_score'] += line[k]*weights[k]
                
            if (line['Bait'], line['Prey']) not in interactions:
                metrics.append(line)
                interactions.add((line['Bait'], line['Prey']))
                
    return metrics

# filter common contaminants
crap = set()
with open('1600105667694_userCrapDB.xls.csv', 'r') as f:
    for n,l in enumerate(f):
        ls = l.strip().split(',')
        if n==0:
            keys = ls
            continue
        line = dict(zip(keys, ls))
        crap.add(line['GENE'])

def get_tt_type(desc):
    if 'IP_' in desc:
        return 'alt'
    elif 'II_' in desc:
        return 'iso'
    elif 'zika' in desc or 'CUFF' in desc:
        return 'zika'
    else:
        return 'ref'

