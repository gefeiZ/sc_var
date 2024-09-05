# package load
import os
import sys
import csv
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scdrs
import sklearn
import mygene
# Request internet connection
mg = mygene.MyGeneInfo()
import pysam
from typing import List, Dict, Tuple
from statsmodels.stats.multitest import multipletests
import matplotlib.transforms as mtrans
import matplotlib.patches as patches
import gseapy as gp
from gseapy import Msigdb
import episcanpy as epi



#########################################################################################
#####                                ANNOTATION Module                             ######
#########################################################################################
   
def read_gwas(gwas_file):

    '''
    Read GWAS data from txt file
    GWAS file  should like 
    chr pos rsids pval
    chr1 100001 rs123 0.01

    if GWAS data from finn database
    USE load_snp_list function
    ''' 
    snp_list=pd.read_csv(gwas_file,sep='\t')
    snp_list['chr']=snp_list['chr'].astype(str)
    snp_list['rsids'] = snp_list['rsids'].astype(str)
    snp_list['pos'] = snp_list['pos'].astype(int)
    snp_list.sort_values(['chr', 'pos'], inplace=True)
    snp_list.reset_index(drop=True, inplace=True)
    return snp_list
    
def load_snp_list(gwas_file):

    '''
    Load GWAS data from finn database
    '''
    
    snp_list = pd.read_csv(gwas_file, 
                            sep='\t',
                            dtype={'#chrom':str,'pos':int,'rsids':str,'pval':float})
    snp_list['chr']='chr'+ snp_list['#chrom']
    snp_list=snp_list.drop(columns=['#chrom'])
    #for rsids column split the data "," to two rows
    snp_list=snp_list.assign(rsids=snp_list['rsids'].str.split(',')).explode('rsids')
    snp_list[['rsids']]=snp_list['rsids'].str.split(',',expand=True)
    #drop row with null rsid
    snp_list=snp_list.dropna(subset=['rsids'],axis=0)
    snp_list=snp_list.dropna(subset=['pval'],axis=0)
    snp_list.sort_values(['chr', 'pos'], inplace=True)
    snp_list.reset_index(drop=True, inplace=True)
    #save snp_list as txt file named by phenotype
    order=['chr','pos','rsids','pval']
    snp_list=snp_list[order]
    snp_list.to_csv(gwas_file,sep='\t',index=False)
    
    return snp_list    


def load_peak_data(peak_file):

    '''
    Load peak to gene file from cicero
    if only have co-accessibility link file from cicero
    Use get_p2g_conn function to get peak to gene file
    '''

    peak_list = pd.read_csv(peak_file,sep='\t')
    peak_list['chr'] = peak_list['chr'].astype(str)
    peak_list['start'] = peak_list['start'] .astype(int)
    peak_list['end'] = peak_list['end'].astype(int)
    peak_list.sort_values(['chr', 'start', 'end'], ignore_index=True, inplace=True)
    peak_list.reset_index(drop=True, inplace=True)

    return peak_list
    
    
def get_p2g_conn(cicero_conn,fdata):
    
    
    '''
    A function help to get the peak-gene conn from cicero_conn and cds fdata

    cicero_conn: output file from cicero conns 

    fdata: from cds fdata 
    Can get from cicero R code:
    after run cicero, get the annoration
    input_cds <- annotate_cds_by_site(input_cds, gene_annotation_sub)
    fdata<-fData(input_cds)
    
    AFTER RUN THIS FUCTION 
    return a txt file with ['gene','chr','start','end'] columns
    which can be used for snp_peak function as peak_list

    '''
    cpeak=pd.read_csv(cicero_conn,sep='\t')
    fdata=pd.read_csv(fdata,sep='\t',index_col=0)
    fdata.reset_index(inplace=True)
    fdata=fdata[['index','gene']]
    conn=cpeak[['Peak1','Peak2']]
    conn['peak1.gene_name']=conn['Peak1'].map(fdata.set_index('index')['gene'])
    conn['peak2.gene_name']=conn['Peak2'].map(fdata.set_index('index')['gene'])
    
    #drop peak1 and peak2 contain chrx and chry
    conn=conn[conn['Peak1'].str.contains('chrX')==False]
    conn=conn[conn['Peak1'].str.contains('chrY')==False]
    conn=conn[conn['Peak2'].str.contains('chrX')==False]
    conn=conn[conn['Peak2'].str.contains('chrY')==False]

    conn=conn[conn['peak1.gene_name'].notnull()|conn['peak2.gene_name'].notnull()]
    conn['gene']=conn['peak1.gene_name'].astype(str)+','+conn['peak2.gene_name'].astype(str)
    #repalce nan, with ' ' in gene
    conn['gene']=conn['gene'].str.replace('nan,','').str.replace(',nan',' ').str.replace('nan','')
    conn['gene']=conn['gene'].str.replace(' ',',')
    conn_filter=conn.assign(gene=conn['gene'].str.split(',')).explode('gene')
    p2=conn_filter[['Peak2','gene']]
    p2.rename(columns={'Peak2':'Peak','gene':'gene'},inplace=True)
    p1=conn_filter[['Peak1','gene']]
    p1.rename(columns={'Peak1':'Peak','gene':'gene'},inplace=True)
    p2g = pd.concat([p1, p2], ignore_index=True)
    p2g=p2g[p2g['gene'].notnull()]
    p2g=p2g[p2g['gene']!='']
    p2g.drop_duplicates(inplace=True)
    p2g['chr']=p2g['Peak'].str.split('-').str[0]
    p2g['start']=p2g['Peak'].str.split('-').str[1]
    p2g['end']=p2g['Peak'].str.split('-').str[2]
    p2g=p2g[['gene','chr','start','end']]
    #get work path and save the result
    work_path=os.getcwd()
    p2g.to_csv(work_path+'/p2g_conn.txt',sep='\t',index=False)
    
    return p2g    
    


def multi_omics_p2g(p2g_file):
    '''
    Read multi omics peak to gene link file
    which is output from signac LinkPeaks() function
    '''
    p2g=pd.read_csv(p2g_file,sep=',')
    p2g=p2g[['gene','peak']]
    p2g['chr']=p2g['peak'].str.split('-').str[0].astype(str)
    p2g['start']=p2g['peak'].str.split('-').str[1].astype(int)
    p2g['end']=p2g['peak'].str.split('-').str[2].astype(int)
    p2g=p2g.drop('peak',axis=1)
    p2g.sort_values(['chr', 'start', 'end'], ignore_index=True, inplace=True)
    p2g.reset_index(drop=True, inplace=True)
    return p2g  







def dmagma(magma_file):

    '''
    load magma annotation file with window size [0,0]

    magma_file: output file from 
    magma --annotate window=0,0 --snp-loc snp_loc.txt --gene-loc gene_loc.txt --out magma.genes.annot

    '''

    with open(magma_file,'r') as file:
        dmagma=file.readlines()
        dmagma=pd.DataFrame(dmagma)
        file.close()
    dmagma=dmagma.drop([0,1])
    dmagma.rename(columns={0:'content'},inplace=True)
    dmagma[['GENE','POS','SNPs']]=dmagma['content'].str.split(n=2,expand=True)
    dmagma.drop(columns=['content'],inplace=True)
    dmagma['SNPs']=dmagma['SNPs'].str.replace('\t',' ')
    dmagma['SNPs']=dmagma['SNPs'].str.split()
    dmagma['SNPs']=dmagma['SNPs'].apply(lambda x: list(set(x)))
    dmagma['SNPs']=dmagma['SNPs'].astype(str)
    dmagma['SNPs']=dmagma['SNPs'].str.replace('\[|\]|\'','')
    dmagma['SNPs'] = dmagma['SNPs'].str.replace('[', '').str.replace(']', '').str.replace("'", '')
    dmagma['SNPs']=dmagma['SNPs'].str.replace(',',' ')
    return dmagma


def snp_peak(peak_list, snp_list):

    '''
    Determine which SNPs overlap with peaks.
    peak_file: peak with co-accessibility score over 0.3(any value)
    snp_List: GWAS SNP list 
    GWAS data should be in txt file with ['chr','pos','rsids','pval'] columns
    '''



    peak_index = 0
    snp_index = 0

    peak_overlaps = list()
    snp_overlaps = list()

    peak_iterator = peak_list.iterrows()
    snp_iterator = snp_list.iterrows()

    peak_index, peak = next(peak_iterator, (None, None))
    snp_index, snp = next(snp_iterator, (None, None))

    # Iterate until either the peaks or SNPs are all considered for overlap
    while peak_index is not None and snp_index is not None:

        if peak.chr == snp.chr:
            if peak.start <= snp.pos <= peak.end:
                peak_overlaps.append(peak_index)
                snp_overlaps.append(snp_index)
                snp_index, snp = next(snp_iterator, (None, None))
            elif snp.pos < peak.start:
                snp_index, snp = next(snp_iterator, (None, None))
            else:
                peak_index, peak = next(peak_iterator, (None, None))
        elif peak.chr < snp.chr:
            peak_index, peak = next(peak_iterator, (None, None))
        else:
            snp_index, snp = next(snp_iterator, (None, None))
    
    snp_list_overlaps = snp_list.drop('chr', axis=1).iloc[snp_overlaps, :]
    snp_list_overlaps.reset_index(drop=True, inplace=True)

    peak_info_overlaps = peak_list.iloc[peak_overlaps, :]
    peak_info_overlaps.reset_index(drop=True, inplace=True)

    overlap_matrix = pd.merge(snp_list_overlaps, peak_info_overlaps, left_index=True, right_index=True)
    
    return overlap_matrix


def gene_corr(gff3_file,peak_file):

    '''
    MAGMA analysis requires gene coordinates
    Extract gene coordinates from gff3 file

    gff3_file:From GENCODE

    '''

    gencode = pd.read_table(gff3_file, comment="#",
                           sep = "\t", names = ['seqname', 'source', 'feature', 'start' , 'end', 'score', 'strand', 'frame', 'attribute'])
    
    
    def gene_info(x):

        g_name = list(filter(lambda x: 'gene_name' in x,  x.split(";")))[0].split("=")[1]
        g_type = list(filter(lambda x: 'gene_type' in x,  x.split(";")))[0].split("=")[1]
        g_leve = int(list(filter(lambda x: 'level' in x,  x.split(";")))[0].split("=")[1])
        return (g_name, g_type, g_leve)
    
    def fetch_gene_coords(g):

        if gencode_genes.index.isin([g]).any(): 
            return gencode_genes.loc[g]['seqname'], gencode_genes.loc[g]['start'], gencode_genes.loc[g]['end']  #gencode_genes.loc[g][['seqname', 'start', 'end']]
        else:
            return "NA", "NA", "NA"
    
    
 
    gencode_genes = gencode[(gencode.feature == "gene")][['seqname', 'start', 'end', 'attribute']].copy().reset_index().drop('index', axis=1)
    gencode_genes["gene_name"], gencode_genes["gene_type"],  gencode_genes["gene_level"] = zip(*gencode_genes.attribute.apply(lambda x: gene_info(x)))
    gencode_genes = gencode_genes.set_index('gene_name')
    
    peak_file['g_chr'], peak_file['g_start'], peak_file['g_end'] = zip(*peak_file['gene'].apply(lambda x: fetch_gene_coords(x)))
    peak_file=peak_file[peak_file['g_chr'].str.contains('NA')==False]
    gene_cor=peak_file[['gene','g_chr','g_start','g_end']]
    gene_cor=gene_cor.drop_duplicates(subset=['gene'],keep='first')
    gene_cor['g_chr']=gene_cor['g_chr'].str.replace('chr','')
    gene_cor['g_chr']=gene_cor['g_chr'].str.replace('X','23')
    gene_cor['g_cor']=gene_cor['g_chr']+':'+gene_cor['g_start'].astype(str)+':'+gene_cor['g_end'].astype(str)
    gene_cor=gene_cor[['gene','g_cor']]
    
    return gene_cor


def merge_annotate(ori_annot,link_annot):

    '''
    OPTIONAL
    For merging any annotation file together 

    '''
    with open(ori_annot,'r') as file:
        magma=file.readlines()
        magma=pd.DataFrame(magma)
        file.close()
    with open(link_annot,'r') as file:
        link=file.readlines()
        link=pd.DataFrame(link)
        file.close()
    magma=magma.drop([0,1])
    magma.rename(columns={0:'content'},inplace=True)
    magma[['GENE','POS','SNPs']]=magma['content'].str.split(n=2,expand=True)
    magma.drop(columns=['content'],inplace=True)
    magma['SNPs']=magma['SNPs'].str.replace('nan',' ').str.replace('\n','').str.replace('\t',' ')
    link.rename(columns={0:'content'},inplace=True)
    link[['GENE','POS','SNPs']]=link['content'].str.split(n=2,expand=True)
    link.drop(columns=['content'],inplace=True)
    link['SNPs']=link['SNPs'].str.replace('nan',' ').str.replace('\n','').str.replace('\t',' ')
    link['SNPs']=link['SNPs'].str.replace('[','').str.replace(']','').str.replace("'",'')
    
    merge_annotate=pd.merge(magma,link,on='GENE',how='outer')
    merge_annotate['POS']=merge_annotate['POS_x'].fillna(merge_annotate['POS_y'])
    merge_annotate['SNPs_x']=merge_annotate['SNPs_x'].astype(str)
    merge_annotate['SNPs_y']=merge_annotate['SNPs_y'].astype(str)
    merge_annotate['SNPs'] = merge_annotate.apply(lambda x: str(x['SNPs_x']) + ' ' + str(x['SNPs_y']), axis=1)
    #remove duplicates in SNPs 
    merge_annotate['SNPs']=merge_annotate.apply(lambda x: set(x['SNPs'].split()), axis=1)
    merge_annotate['SNPs']=merge_annotate['SNPs'].astype(str)
    merge_annotate['SNPs']=merge_annotate['SNPs'].str.replace('nan',' ').str.replace('\n','').str.replace('\t',' ')
    merge_annotate['SNPs']=merge_annotate['SNPs'].str.replace("'",'')
    merge_annotate['SNPs']=merge_annotate['SNPs'].str.replace('{','').str.replace('}','').str.replace(",",' ')
    merge_annotate['SNPs'] = merge_annotate['SNPs'].drop_duplicates()


    merge_annotate=merge_annotate[['GENE','POS','SNPs']]
    merge_annotate.to_csv(link_annot, sep='\t', index=False, header=False)

    
    return merge_annotate





def annotate(overlap_matrix,gene_cor,dmagma_file,disease_name):
    '''
    Annotate GWAS SNPs with co_accessibility peaks with gene information


    overlap_matrix: output from dis_peak function
    gene_cor: output from gene_corr function
    dmagma_file: output from dmagma function
    disease_name: disease name interested from GWAS
    '''
    
    gene_list=overlap_matrix.groupby('gene')['rsids'].apply(list).reset_index(name='snps')
    gene_list=pd.merge(gene_list,gene_cor,on='gene',how='left')
    gene_id=mg.querymany(gene_list['gene'], scopes='symbol', fields='entrezgene', species='human', as_dataframe=True)['entrezgene']
    gene_id=gene_id.to_frame().reset_index()
    gene_id.dropna(inplace=True)
    gene_id=gene_id.rename(columns={'query':'gene'})
    convert=pd.merge(gene_list,gene_id,on='gene',how='left')
    convert.drop(columns=['gene'],inplace=True)
    convert['snps']=convert['snps'].astype(str)
    convert['snps']=convert['snps'].str.replace("nan","")
    convert['snps']=convert['snps'].str.replace(',',' ')
    convert['snps']=convert['snps'].str.replace('[','')
    convert['snps']=convert['snps'].str.replace(']','')
    convert['snps']=convert['snps'].str.replace("'","")
    convert.dropna(inplace=True)
    convert.dropna(subset=['g_cor'],inplace=True)
    convert.rename(columns={'entrezgene':'GENE'},inplace=True)

    cmagma=pd.merge(dmagma_file,convert,on='GENE',how='outer')
    cmagma['POS']=cmagma['POS'].fillna(cmagma['g_cor'])
    cmagma['SNPs']=cmagma['SNPs'].astype(str)
    cmagma['snps']=cmagma['snps'].astype(str)
    cmagma['SNPs'] = cmagma.apply(lambda x: str(x['SNPs']) + ' ' + str(x['snps']), axis=1)
    cmagma['SNPs']=cmagma['SNPs'].str.replace('nan',' ')
    cmagma['SNPs']=cmagma['SNPs'].str.replace('[','').str.replace(']','').str.replace("'",'')
    cmagma=cmagma[['GENE','POS','SNPs']]
    cmagma.to_csv(disease_name+'scemagma.genes.annot',index=False,header=False,sep='\t',quoting=csv.QUOTE_NONE)
    return cmagma



def save_dict_gs(gs_path: str, dict_gs: dict) -> None:
    """
    Save dict_gs to gs file (.gs file).
    df_gs: Dict = {
        "TRAIT": [],
        "GENESET": [],}
    """
    df_gs: Dict = {
        "TRAIT": [],
        "GENESET": [],
    }
    for trait in dict_gs:
        df_gs["TRAIT"].append(trait)
        if isinstance(dict_gs[trait], tuple):
            df_gs["GENESET"].append(
                ",".join([str(g) + ":" + str(w) for g, w in zip(*dict_gs[trait])])
            )
        else:
            df_gs["GENESET"].append(",".join(dict_gs[trait]))
    pd.DataFrame(df_gs).to_csv(gs_path, sep="\t", index=False)
    

def save_gs(gene_file,disease_name,gs_path):

    '''
    OPTIONAL
    Directly save scemagma gene analysis results into .gs file

    gene_file: gene analysis results from scemagma  rename to txt file
    disease_name: disease name interested from GWAS
    gs_path: output path for .gs file

    '''
    gene_file=pd.read_csv(gene_file,sep=r'\s+')
    gene_id=mg.querymany(gene_file['GENE'], scopes='entrezgene', fields='symbol', species='human', as_dataframe=True)['symbol']
    gene_id=pd.DataFrame(gene_id)
    gene_id=gene_id.reset_index('query')
    gene_id=gene_id.rename(columns={'query':'GENE'})
    gene_id['GENE']=gene_id['GENE'].astype('str')
    gene_file['GENE']=gene_file['GENE'].astype('str')
    gene_file=pd.merge(gene_file,gene_id,on='GENE',how='left')
    gene_file=gene_file[gene_file['P']<0.05]
    gene_file=gene_file.sort_values(by='P')
    gene_file=gene_file.drop_duplicates(subset='symbol',keep='first')
    gene_file=gene_file[['symbol','ZSTAT']]
    dict_gs = {disease_name: (gene_file['symbol'].tolist(), gene_file['ZSTAT'].tolist())}
    df_gs: dict = {
        "TRAIT": [],
        "GENESET": [],
    }
    for trait in dict_gs:
        df_gs["TRAIT"].append(trait)
        if isinstance(dict_gs[trait], tuple):
            df_gs["GENESET"].append(
                ",".join([str(g) + ":" + str(w) for g, w in zip(*dict_gs[trait])])
            )
        else:
            df_gs["GENESET"].append(",".join(dict_gs[trait]))
    pd.DataFrame(df_gs).to_csv(gs_path, sep="\t", index=False)






#########################################################################################
#####                                SCORING                                       ######
#########################################################################################

def get_peak_weights(overlap_matrix):

    '''
    For each peak, determine the lowest p-value of all SNPs that overlap with it.
    '''

    peak_weights = overlap_matrix.groupby(['chr', 'start', 'end']).agg({'pval': 'min'})
    #if min p is 0, replace it with 1e-10
    peak_weights['pval']=peak_weights['pval'].replace(0,1e-10)
    overlap_matrix['zscore']= np.abs(scipy.stats.norm.ppf(overlap_matrix['pval']/ 2))
    #keep zcore corresponding to min p
    peak_weights=pd.merge(peak_weights,overlap_matrix[['chr','start','end','zscore']],on=['chr','start','end'],how='left')
    peak_weights.reset_index(inplace=True)
    peak_weights.rename({'zscore': 'weight'}, axis=1, inplace=True)
    peak_weights['GENE']=peak_weights['chr'].astype(str)+'-'+peak_weights['start'].astype(str)+'-'+peak_weights['end'].astype(str)

    peak_weights.sort_values(['weight'],ascending=False,inplace=True)
    peak_weights.drop_duplicates(subset=['GENE'],keep='first',inplace=True)
    peak_weights=peak_weights[['GENE','weight']]
    return peak_weights

def load_genes(gene_file):
    gene_file=pd.read_csv(gene_file,sep=r'\s+')
    gene_id=mg.querymany(gene_file['GENE'], scopes='entrezgene', fields='symbol', species='human', as_dataframe=True)['symbol']
    gene_id=pd.DataFrame(gene_id)
    gene_id=gene_id.reset_index('query')
    gene_id=gene_id.rename(columns={'query':'GENE'})
    gene_id['GENE']=gene_id['GENE'].astype('str')
    gene_file['GENE']=gene_file['GENE'].astype('str')
    gene_file=pd.merge(gene_file,gene_id,on='GENE',how='left')
    return gene_file

def drop_dup(gene_file):
    gene_file=gene_file.sort_values(by='P')
    gene_file=gene_file.drop_duplicates(subset='symbol',keep='first')
    return gene_file


def scedrs_atac(adata,overlap_matrix,cmagma_result,disease_name):

    '''
    adata: AnnData object
    
    overlap_matrix: output from snp_peak function
    For each peak, determine the lowest p-value of all SNPs that overlap with it.
    peak_weights can also get from get_peak_weights function 
    
    scemagma_result: output from scemagma pipline gene analysis results
    see /magma \ --bfile /g1000_eur \--pval {gwas} use='rsids,pval' N=n\
    --gene-annot {output from annotate fuction *.scemagma.genes.annot} \--out outfile
    
    disease_name: disease name interested from GWAS

    '''
    cmagma_gene=load_genes(cmagma_result)
    cmagma_gene=drop_dup(cmagma_gene)
    cmagma_gene_sig=cmagma_gene[cmagma_gene['P']<0.05]
    overlap_matrix=overlap_matrix[overlap_matrix['gene'].isin(cmagma_gene_sig['symbol'])]
    
    peak_weights = get_peak_weights(overlap_matrix)
    peak_weights=peak_weights.rename(columns={'weight':disease_name})
    dict_gs = {disease_name: (peak_weights['GENE'].tolist(), peak_weights[disease_name].tolist())}
    dict_df_score = dict()
    scdrs.preprocess(adata,  n_mean_bin=20, n_var_bin=20, copy=False)
    for trait in dict_gs:
        gene_list, gene_weights = dict_gs[trait]
        dict_df_score[trait] = scdrs.score_cell(data=adata,
        gene_list=gene_list,
        gene_weight=gene_weights,
        ctrl_match_key="mean_var",
        n_ctrl=1000,
        weight_opt="vs",
        return_ctrl_raw_score=False,
        return_ctrl_norm_score=True,
        verbose=False,
    )
    return dict_df_score


def load_gs(gs_path: str) -> dict:
    """
    Load gene set file (.gs file).
    """
    df_gs = pd.read_csv(gs_path, sep="\t")
    dict_gs = {}
    for _, row in df_gs.iterrows():
        trait = row["TRAIT"]
        geneset = row["GENESET"].split(",")
        geneset = {g.split(":")[0]: float(g.split(":")[1]) for g in geneset}
        dict_gs[trait] = geneset
    return dict_gs

def scedrs_rna(adata,gs_path):

    '''
    adata: AnnData object

    gs_path: output from scemagma pipline gene analysis results
    or any gene set you interested in .gs format 
    use save_gs function to save gene set file as .gs format
    
    disease_name: disease name interested from GWAS

    '''
    df_gs = load_gs(gs_path)
        
    dict_df_score = dict()
    scdrs.preprocess(adata,  n_mean_bin=20, n_var_bin=20, copy=False)
    for trait in df_gs:
        gene_list, gene_weights = df_gs[trait]
        dict_df_score[trait] = scdrs.score_cell(data=adata,
        gene_list=gene_list,
        gene_weight=gene_weights,
        ctrl_match_key="mean_var",
        n_ctrl=1000,
        weight_opt="vs",
        return_ctrl_raw_score=False,
        return_ctrl_norm_score=True,
        verbose=False,
    )
    return dict_df_score

        
#########################################################################################
#####                          Statistical Analysis                                ######
#########################################################################################

def table_output(adata,overlap_matrix,gs_file,anno,omic,group):

    """
    adata: AnnData
    overlap_matrix: pd.DataFrame from scv.snp_peak output
    gs_file: str, path to the risk gene .gs file
    anno: anno = scv.annotate()
    omic: ATAC or RNA
    group: The cell type you focus on (Typically the cell type have significant association with the disease)
    """

    if omic=='ATAC':
        epi.tl.rank_features(adata, 'celltype', omic='ATAC')
        diff_peaks =adata.uns['rank_features_groups']['names']
        diff_peaks = pd.DataFrame(diff_peaks)
        overlap_matrix['peak']=overlap_matrix['chr'].astype(str)+'-'+overlap_matrix['start'].astype(str)+'-'+overlap_matrix['end'].astype(str)
        gs_df=load_gs(gs_file)
        gs_df=pd.DataFrame(gs_df)
        group_result=diff_peaks[[group]][diff_peaks[[group]][group].isin(gs_df.index)]
        results=overlap_matrix[overlap_matrix['peak'].isin(group_result[group])]
        results=results[['peak','gene']]
        trans=mg.querymany(anno['GENE'], scopes='entrezgene', fields='symbol', species='human', as_dataframe=True)
        gene_id=trans['symbol']
        gene_id=gene_id.to_frame().reset_index()
        gene_id.dropna(inplace=True)
        gene_id=gene_id.rename(columns={'query':'GENE'})
        convert=pd.merge(anno,gene_id,on='GENE',how='left')
        convert=convert.rename(columns={'symbol':'gene'})
        convert=convert[['gene','SNPs']]
        output_list=pd.merge(results,convert,on='gene',how='left')
        output=output_list.groupby('peak').agg('first')
        output.to_csv(group+'_atac_output.csv',sep='\t',header=True,index=True)
    
    if omic=='RNA':
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, "celltype", method="wilcoxon",use_raw=False)
        diff_genes=pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
        #get top 100 genes
        diff_genes=diff_genes.iloc[:100,:]
        diff_genes=pd.DataFrame(diff_genes)
        gs_df=load_gs(gs_file)
        gs_df=pd.DataFrame(gs_df)
        group_result=diff_genes[[group]][diff_genes[[group]][group].isin(gs_df.index)]
        results=overlap_matrix[overlap_matrix['gene'].isin(group_result[group])]
        results=results[['gene','peak']]
        trans=mg.querymany(anno['GENE'], scopes='entrezgene', fields='symbol', species='human', as_dataframe=True)
        gene_id=trans['symbol']
        gene_id=gene_id.to_frame().reset_index()
        gene_id.dropna(inplace=True)
        gene_id=gene_id.rename(columns={'query':'GENE'})
        convert=pd.merge(anno,gene_id,on='GENE',how='left')
        convert=convert.rename(columns={'symbol':'gene'})
        
        convert=convert[['gene','SNPs']]
        output_list=pd.merge(results,convert,on='gene',how='left')
        enr = gp.enrichr(gene_list=list(output_list['gene'].unique()),
                 gene_sets=['MSigDB_Hallmark_2020','KEGG_2021_Human'],
                 organism='human', 
                 outdir=None,
                )
        enr.results.to_csv(group+'_GO_output.csv',sep='\t',index=False)
        output=output_list
        output.to_csv(group+'_rna_output.csv',sep='\t',header=True,index=True)
    


    return output




def stat_analysis(
    adata,
    df_full_score,
    group_cols,
    fdr_thresholds=[0.05, 0.1, 0.2]):

    cell_list = sorted(set(adata.obs_names) & set(df_full_score.index))
    control_list = [x for x in df_full_score.columns if x.startswith("ctrl_norm_score")]
    n_ctrl = len(control_list)
    df_reg = adata.obs.loc[cell_list, group_cols].copy()
    df_reg = df_reg.join(
        df_full_score.loc[cell_list, ["norm_score"] + control_list + ["pval"]]
    )

    # Group-level analysis; dict_df_res : group_col -> df_res
    dict_df_res = {}
    for group_col in group_cols:
        group_list = sorted(set(adata.obs[group_col]))
        res_cols = [
            "n_cell",
            "n_ctrl",
            "assoc_mcp",
            "assoc_mcz"]
        for fdr_threshold in fdr_thresholds:
            res_cols.append(f"n_fdr_{fdr_threshold}")

        df_res = pd.DataFrame(index=group_list, columns=res_cols)
        df_res.index.name = "group"
        pvals = df_reg["pval"].values
        pvals[pvals == 0] = 1e-10 
        df_fdr = pd.DataFrame(
            {"fdr": multipletests(df_reg["pval"].values, method="fdr_bh")[1]},
            index=df_reg.index,)


        for group in group_list:
            group_cell_list = list(df_reg.index[df_reg[group_col] == group])
            # Basic info
            df_res.loc[group, ["n_cell", "n_ctrl"]] = [len(group_cell_list), n_ctrl]

            # Number of FDR < fdr_threshold cells in each group
            for fdr_threshold in fdr_thresholds:
                df_res.loc[group, f"n_fdr_{fdr_threshold}"] = (
                    df_fdr.loc[group_cell_list, "fdr"].values < fdr_threshold
                ).sum()

        # Association
        for group in group_list:
            group_cell_list = list(df_reg.index[df_reg[group_col] == group])
            score_q95 = np.quantile(df_reg.loc[group_cell_list, "norm_score"], 0.95)
            v_ctrl_score_q95 = np.quantile(
                df_reg.loc[group_cell_list, control_list], 0.95, axis=0
            )
            mc_p = ((v_ctrl_score_q95 >= score_q95).sum() + 1) / (
                v_ctrl_score_q95.shape[0] + 1
            )
            mc_z = (score_q95 - v_ctrl_score_q95.mean()) / v_ctrl_score_q95.std()
            df_res.loc[group, ["assoc_mcp", "assoc_mcz"]] = [mc_p, mc_z]
            
        dict_df_res[group_col] = df_res      

    return dict_df_res



#########################################################################################
#####                                Visualization                                 ######
#########################################################################################

from statsmodels.stats.multitest import multipletests
import matplotlib.transforms as mtrans
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.pyplot as plt



def plot_group_stats(dict_df_stats=None):

    
    trait_list = list(dict_df_stats.keys())
    # compile df_fdr_prop, df_assoc_fdr, df_hetero_fdr from dict_df_stats
    df_fdr_prop = pd.concat(
        [
            #-np.log10(dict_df_stats[trait]["assoc_mcp"])+1 
            #q10
            dict_df_stats[trait]['q10']
            for trait in trait_list
        ],
        axis=1,
    ).T


    df_assoc_fdr = pd.concat(
        [dict_df_stats[trait]["assoc_mcp"] for trait in trait_list], axis=1
    ).T

    df_assoc_fdr = pd.DataFrame(
        df_assoc_fdr.values,
        index=df_assoc_fdr.index,
        columns=df_assoc_fdr.columns,
    )


    df_hetero_fdr = pd.concat(
        [dict_df_stats[trait]["assoc_mcp"] for trait in trait_list], axis=1
    ).T
    df_hetero_fdr = pd.DataFrame(
       df_hetero_fdr.values,
            index=df_hetero_fdr.index,
            columns=df_hetero_fdr.columns,
        )
    
    
    df_fdr_prop.index = trait_list
 
    df_assoc_fdr.index = trait_list
 
    df_hetero_fdr.index = trait_list
  

    df_hetero_fdr = df_hetero_fdr.applymap(lambda x: "" if x < 0.05 else "")
    df_hetero_fdr[df_assoc_fdr > 0.1] = ""
   
    fig, ax = plot_heatmap(
        df_fdr_prop,
        squaresize=40,
        heatmap_annot=df_hetero_fdr,
        heatmap_annot_kws={"color": "blue", "size": 8},
        heatmap_cbar_kws=dict(
            use_gridspec=False, location="top", fraction=0.01, pad=0.3, drawedges=True),
        heatmap_vmin=0,
        heatmap_vmax=4,
        colormap_n_bin=8,
    )

    small_squares(
        ax,
        pos=[(y, x) for x, y in zip(*np.where(df_assoc_fdr < 0.01))],
        size=0.6,
        linewidth=0.5,
    )

    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=4)

    cb.ax.set_title("Risk Score", fontsize=8)
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(1)

    plt.tight_layout()

def discrete_cmap(N, base_cmap=None, start_white=True):
    base = plt.colormaps.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    if start_white:
        color_list[0, :] = 1.0
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)



def plot_heatmap(
    df,
    dpi=150,
    squaresize=10,
    heatmap_annot=None,
    heatmap_annot_kws={"color": "black", "size": 10},
    heatmap_linewidths=0.5,
    heatmap_linecolor="gray",
    heatmap_xticklabels=True,
    heatmap_yticklabels=True,
    heatmap_cbar=True,
    heatmap_cbar_kws=dict(use_gridspec=False, location="top", fraction=0.05, pad=0.1),
    heatmap_vmin=0,
    heatmap_vmax=4.0,
    xticklabels_rotation=90,
    colormap_n_bin=8,
):
    #figwidth = df.shape[1] * (squaresize) / float(dpi)
    #figheight = df.shape[0] * squaresize / float(dpi)
    figwidth = 6
    figheight = 6
    fig, ax = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor("silver")
    ax.invert_yaxis()
    #ax.yaxis.set_label_position("right")
    #ax.yaxis.tick_right()
    sns.heatmap(
        df,
        annot=heatmap_annot,
        annot_kws=heatmap_annot_kws,
        fmt="",
        cmap=discrete_cmap(colormap_n_bin, "YlOrBr"),
        linewidths=heatmap_linewidths,
        linecolor=heatmap_linecolor,
        square=True,
        ax=ax,
        xticklabels=heatmap_xticklabels,
        yticklabels=heatmap_yticklabels,
        cbar=heatmap_cbar,
        cbar_kws=heatmap_cbar_kws,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    plt.yticks(fontsize=8)

    for tick_label in ax.axes.get_yticklabels():
        tick_label.set_color("black")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=xticklabels_rotation,
        va="top",
        ha="right",
        fontsize=8,
    )
    ax.tick_params(left=False, bottom=False, pad=-2)
    trans = mtrans.Affine2D().translate(5, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)
    return fig, ax
    
def small_squares(ax, pos, size=1, linewidth=0.8):
    """
    Draw many small squares on ax, given the positions of
    these squares.

    """
    for xy in pos:
        x, y = xy
        margin = (1 - size) / 2
        rect = patches.Rectangle(
            (x + margin, y + margin),
            size,
            size,
            linewidth=linewidth,
            edgecolor="black",
            facecolor="none",
            zorder=15,
        )
        ax.add_patch(rect)