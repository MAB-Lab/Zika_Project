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
from pycompomics.pycompomics.hierarchical_parser import *
from pycompomics.pycompomics import SearchGUI, PeptideShaker
from pymsconvert.pymsconvert import Converter
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
from scipy.stats import hypergeom

uniprot_genes = {}
with open('uniprot_HGNC.tab', 'r') as f:
    for n,l in enumerate(f):
        ls = l.strip().split('\t')
        if n==0:
            keys = ls
            continue
        line = dict(zip(keys, ls))
        uniprot_genes[line['Entry']] = line['Gene names']


# ensp_genes dictionary
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

        
OP_genes = {}
with open('human-openprot-r1_5-refprots+altprots+isoforms-+uniprot2019_03_01.tsv', 'r') as f:
    for n,l in enumerate(f):
        if l[0]=='#': continue
        ls = l.strip().split('\t')    
        if n==1:
            keys = ls
            continue
        line=dict(zip(keys, ls))    
        acc = line['protein accession numbers'].split('.')[0]
        gene = line['gene symbol']
        if any(x in acc for x in ['IP_', 'II_']):
            gene = '|'.join([gene, acc])
        OP_genes[acc] = gene
        
prot_genes = {}
prot_genes.update(OP_genes)
prot_genes.update(ensp_genes)
prot_genes.update(uniprot_genes)

def get_tt_type(prot_acc):
    if 'IP_' in prot_acc:
        return 'AltProt'
    elif 'II_' in prot_acc:
        return 'Novel Isoform'
    elif 'zika' in prot_acc:
        return 'zika'
    else:
        return 'Reference'

def get_prey_gene(p_acc):
    p_acc_nover = p_acc.split('.')[0]
    if 'CUFF' in p_acc:
        return p_acc
    elif p_acc_nover in prot_genes and prot_genes[p_acc_nover]:
        return prot_genes[p_acc_nover].split()[0]
    else:
        print(p_acc)
        return None

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
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}
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

# Load supplementary data
denv_files = []
denv_bait_files = {}
with open('DENV/experiment.csv', 'r') as f:
    for n,l in enumerate(f):
        if n<13:continue
        ls = l.strip().split('", "')
        try:
            fname, treatment = ls
            denv_files.append(fname[1:])
            t = treatment.split(', ')
            bait_name = t[2].replace('"', '')
            if 'S135A' in bait_name:
                continue
            denv_bait_files[fname[1:].replace('.raw', '')] = bait_name
        except:
            continue

uganda_files = []
zikv_bait_files = {}
with open('zika_raw/experiment.csv', 'r') as f:
    for n,l in enumerate(f):
        if n<13:continue
        ls = l.strip().split('", "')
        try:
            fname, treatment = ls
            if 'Uganda' in treatment:
                uganda_files.append(fname[1:])
                t = treatment.split(', ')
                bait_name = t[2].replace('"', '')
                if 'S135A' in bait_name:
                    continue
                zikv_bait_files[fname[1:].replace('.raw', '')] = t[2]
        except:
            continue

table_s2xl = pyexcel.load('NIHMS1515954-supplement-8.xlsx')
table_s2 = []
for n,row in enumerate(table_s2xl):
    if n==0:continue
    if n==1:
        keys = row
        continue
    table_s2.append(dict(zip(keys, row)))

s2_interactions = []
for bench in table_s2:
    bait = bench['Top Scoring BAIT']
    label = bench['Benchmark']
    if bait in denv_bait_files:
        bait = denv_bait_files[bait]
    if type(bait) == list:
        for b in bait:
            s2_interactions.append((b, bench['PREY'], label))
        continue
    s2_interactions.append((bait, bench['PREY'], label))
s2_interactions[:10]

bench_prey_set = {}
for bait, grp in itt.groupby(sorted(s2_interactions), key=lambda x: x[0]):
    bench_prey_set[bait] = set([x[1] for x in grp])

prot_lens = {}
for prot in SeqIO.parse('zika_OP14_RNAseq_custom.fasta', 'fasta'):
    acc = prot.name.split('|')[0]
    prot_lens[acc] = len(prot.seq)


for prot in SeqIO.parse('ZIKVug_swissprot.fasta', 'fasta'):
    acc = prot.name.split('|')[1]
    prot_lens[acc] = len(prot.seq)


sup7 = pyexcel.load('NIHMS1515954-supplement-7.xlsx')
keys = sup7.array[1]
for l in sup7.array[2:]:
    line = dict(zip(keys, l))
    if 'ZIKVug' in line['Bait']:
        strain, prot_acc = line['Bait'].split()
        prot_lens[prot_acc] = len(line['Protein Sequence'][:-1])

sup10 = pyexcel.load('NIHMS1515954-supplement-10.xlsx')

lit_zikv_net = set()
for n,x in enumerate(sup10):
    if n==0:continue
    if n==1:
        keys = x
        continue
    line = dict(zip(keys, x))
    if line['ZIKV-Human M>0.72(fp) or 0.69 (ug)'] == True and 'ZIKVug' in line['Bait']:
        bait = line['Bait'].replace('ZIKVug ', '')
        lit_zikv_net.add((bait, line['Prey'], line['Gene.names']))
zikv_lit_preys = set(x[1] for x in lit_zikv_net)        
lit_zikv_net = [(x[0], get_prey_gene(x[1])) for x in lit_zikv_net if x[1] in uniprot_genes]
lit_zikv_net = set(lit_zikv_net)
zikv_lit_prey_genes = set(x[1] for x in lit_zikv_net)

#genes trxp dict
fpath = '/home/xroucou_group/echange_de_fichiers/Zika_Marie_Seb/gencode.v32.primary_assembly.annotation.gtf'
keys = ['chrom', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'desc']
trxps_by_gene = {}
test = []
with open(fpath, 'r') as f:
    for n,l in enumerate(f):
        if l[0]=='#':continue
        ls = l.strip().split('\t')
        line = dict(zip(keys, ls))
        if n<100:
            test.append(line)
        if line['feature'] != 'transcript': continue
        gene_name = [x.split()[1].replace('"', '') for x in line['desc'].split('; ') if 'gene_name' in x][0]
        trxp_acc = [x.split()[1].replace('"', '') for x in line['desc'].split('; ') if 'transcript_id' in x][0]
        if gene_name not in trxps_by_gene:
            trxps_by_gene[gene_name] = [trxp_acc,]
        else:
            trxps_by_gene[gene_name].append(trxp_acc)

# dictionary of all OpenProt 1.5 proteins by transcript
enst_trxps = []
all_prots_by_trxp = {}
with open('human-openprot-r1_5-refprots+altprots+isoforms-+uniprot2019_03_01.tsv', 'r') as f:
    for n,l in enumerate(f):
        if l[0]=='#': continue
        ls = l.strip().split('\t')    
        if n==1:
            keys = ls
        line=dict(zip(keys, ls))
        trxp_acc = line['transcript accession'].split('.')[0]
        if 'ENST' not in trxp_acc: continue        
        if trxp_acc not in all_prots_by_trxp:
            all_prots_by_trxp[trxp_acc] = [line['protein accession numbers']]
        else:
            all_prots_by_trxp[trxp_acc].append(line['protein accession numbers'])
        if line['protein type'] != 'RefProt':continue
        uniprot_acc = [x.split('-')[0] for x in line['protein accession (others)'].split(';') if 'ENSP' not in x and '_' not in x]
        if not uniprot_acc: continue
        enst_trxps.append((uniprot_acc[0], trxp_acc))

# ENST trxps for uniprot 
uniprot_enst_trxps = {}
for prot_acc, trxps in itt.groupby(sorted(enst_trxps), key=lambda x: x[0]):
    uniprot_enst_trxps[prot_acc] = list(set(x[1] for x in trxps))

# Uniprot ENST dict from Ensembl BioMart (Downloaded Feb 2020)
with open('uniprot_enst.txt', 'r') as f:
    for n, l in enumerate(f):
        ls = l.strip().split('\t')
        if len(ls)<2: continue
        enst, uniprot = ls
        if uniprot in uniprot_enst_trxps:
            if enst not in uniprot_enst_trxps[uniprot]:
                uniprot_enst_trxps[uniprot].append(enst)
        else:
            uniprot_enst_trxps[uniprot] = [enst, ]

def get_psm_counts(shaker_report, bait=None):
    prots = parse(shaker_report)
    psms = []
    for prot_g in prots:
        prot_acc = prot_g.accession.split('|')[0]
        prot_acc = prot_acc.replace('ZIKVug_', '')
        if ('IP_' in prot_acc or 'II_' in prot_acc) and prot_g.related_proteins:
            alt_pep_only = True
            for p in prot_g.related_proteins:
                if 'IP_' not in p and 'II_' not in p:
                    alt_pep_only = False
            if not alt_pep_only:
                continue

        if bait is not None and bait in bench_prey_set and prot_acc not in bench_prey_set[bait]:
            for p in prot_g.related_proteins:
                if p in bench_prey_set[bait]:
                    prot_acc = p
            
        for pep in prot_g.peptide_rows:
            for psm in pep.psm_rows:
                if psm.validation in ('Confident', 'Very Confident', 'Doubtful'):
                    psms.append((prot_acc, psm.spectrum_title))
    prot_psms = []
    for prot_acc, spectra in itt.groupby(sorted(psms), key=lambda x: x[0]):
        prot_len = 1 if prot_acc not in prot_lens else prot_lens[prot_acc]
        prot_psms.append((exp_name, prot_acc, len(list(spectra)), prot_len))  # prot_lens[prot_acc]
    return prot_psms

def count_mgf_spectra(fpath):
    n_specs = 0
    with open(fpath, 'r') as f:
        for l in f:
            if 'BEGIN ION' in l:
                n_specs += 1
    return n_specs

