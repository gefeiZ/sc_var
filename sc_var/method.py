# package load
import os
import sys
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




#########################################################################################
#####                                ANNOTATION                                    ######
#########################################################################################


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
    cmagma=cmagma[['GENE','POS','SNPs']]

    cmagma_annot=open(disease_name+'cmagma.genes.annot','w')
    cmagma_annot.write(convert.to_string(index=False,header=False))
    cmagma_annot.close()
    
    return cmagma





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
    magma.rename(columns={0:'content'},inplace=True)
    magma[['GENE','POS','SNPs']]=magma['content'].str.split(n=2,expand=True)
    magma.drop(columns=['content'],inplace=True)
    link.rename(columns={0:'content'},inplace=True)
    link[['GENE','POS','SNPs']]=link['content'].str.split(n=2,expand=True)
    link.drop(columns=['content'],inplace=True)
    merge_annotate=pd.merge(magma,link,on='GENE',how='outer')
    
    merge_annotate['POS']=merge_annotate['POS_x'].fillna(merge_annotate['POS_y'])
    
    merge_annotate['SNPs_x']=merge_annotate['SNPs_x'].astype(str)
    merge_annotate['SNPs_y']=merge_annotate['SNPs_y'].astype(str)
    merge_annotate['SNPs'] = merge_annotate.apply(lambda x: str(x['SNPs_x']) + ' ' + str(x['SNPs_y']), axis=1)
    merge_annotate['SNPs']=merge_annotate['SNPs'].str.replace('nan',' ')
    merge_annotate=merge_annotate[['GENE','POS','SNPs']]
    output=open('merge_annotate.genes.annot','w')
    output.write(merge_annotate.to_string(index=False,header=False))
    output.close()
    
    return merge_annotate




def save_gs(gene_file,disease_name,gs_path):

    '''
    OPTIONAL
    Save gene analysis results as .gs file

    gene_file: gene analysis results from cmagma  rename to txt file
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
                ",".join([g + ":" + str(w) for g, w in zip(*dict_gs[trait])])
            )
        else:
            df_gs["GENESET"].append(",".join(dict_gs[trait]))
    pd.DataFrame(df_gs).to_csv(gs_path, sep="\t", index=False)






#########################################################################################
#####                                SCORING                                       ######
#########################################################################################

def get_peak_weights(overlap_matrix):

    '''
    OPTIONAL
    For each peak, determine the lowest p-value of all SNPs that overlap with it.
    '''

    peak_weights = overlap_matrix.groupby(['chr', 'start', 'end']).agg({'pval': 'min'})
    overlap_matrix['zscore']= np.abs(scipy.stats.norm.ppf(overlap_matrix['pval']/ 2))
    #keep zcore corresponding to min p
    peak_weights=pd.merge(peak_weights,overlap_matrix[['chr','start','end','zscore']],on=['chr','start','end'],how='left')
    peak_weights.reset_index(inplace=True)
    peak_weights.rename({'zscore': 'weight'}, axis=1, inplace=True)
    peak_weights['GENE']='chr'+peak_weights['chr'].astype(str)+'-'+peak_weights['start'].astype(str)+'-'+peak_weights['end'].astype(str)

    peak_weights.sort_values(['weight'],ascending=False,inplace=True)
    peak_weights.drop_duplicates(subset=['GENE'],keep='first',inplace=True)
    peak_weights=peak_weights[['GENE','weight']]
    return peak_weights






def scads(adata,overlap_matrix,disease_name):

    '''
    adata: AnnData object
    For each peak, determine the lowest p-value of all SNPs that overlap with it.
    peak_weights can also get from get_peak_weights function 
    disease_name: disease name interested from GWAS

    '''

    peak_weights = overlap_matrix.groupby(['chr', 'start', 'end']).agg({'pval': 'min'})
    overlap_matrix['zscore']= np.abs(scipy.stats.norm.ppf(overlap_matrix['pval']/ 2))
    #keep zcore corresponding to min p
    peak_weights=pd.merge(peak_weights,overlap_matrix[['chr','start','end','zscore']],on=['chr','start','end'],how='left')
    peak_weights.reset_index(inplace=True)
    peak_weights.rename({'zscore': 'weight'}, axis=1, inplace=True)
    peak_weights['GENE']='chr'+peak_weights['chr'].astype(str)+'-'+peak_weights['start'].astype(str)+'-'+peak_weights['end'].astype(str)

    peak_weights.sort_values(['weight'],ascending=False,inplace=True)
    peak_weights.drop_duplicates(subset=['GENE'],keep='first',inplace=True)
    peak_weights=peak_weights[['GENE','weight']]

    dict_gs = {disease_name: (peak_weights['GENE'].tolist(), peak_weights['weight'].tolist())}
    dict_df_score = dict()
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


        
#########################################################################################
#####                          Statistical Analysis                                ######
#########################################################################################


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

        df_fdr = pd.DataFrame(
            {"fdr": multipletests(df_reg["pval"].values, method="fdr_bh")[1]},
            index=df_reg.index,
        )

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

    return dict_df_res



#########################################################################################
#####                                Visualization                                 ######
#########################################################################################



def plot_heatmap(
    df,
    dpi=150,
    squaresize=20,
    heatmap_annot=None,
    heatmap_annot_kws={"color": "black", "size": 4},
    heatmap_linewidths=0.5,
    heatmap_linecolor="gray",
    heatmap_xticklabels=True,
    heatmap_yticklabels=True,
    heatmap_cbar=True,
    heatmap_cbar_kws=dict(use_gridspec=False, location="top", fraction=0.03, pad=0.01),
    heatmap_vmin=0.0,
    heatmap_vmax=1.0,
    xticklabels_rotation=90,
    colormap_n_bin=10,
):
    figwidth = df.shape[1] * squaresize / float(dpi)
    figheight = df.shape[0] * squaresize / float(dpi)
    fig, ax = plt.subplots(1, figsize=(figwidth, figheight), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor("silver")
    sns.heatmap(
        df,
        annot=heatmap_annot,
        annot_kws=heatmap_annot_kws,
        fmt="",
        cmap=discrete_cmap(colormap_n_bin, "RdPu"),
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


def discrete_cmap(N, base_cmap=None, start_white=True):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    if start_white:
        color_list[0, :] = 1.0
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def small_squares(ax, pos, size=1, linewidth=0.8):

    for xy in pos:
        x, y = xy
        margin = (1 - size) / 2
        rect = patches.Rectangle(
            (x + margin, y + margin),
            size,
            size,
            linewidth=linewidth,
            ls='--',
            edgecolor="black",
            facecolor="none",
            zorder=40,
        )
        ax.add_patch(rect)




def plot_stat(dict_df_stats=None):

    
    trait_list = list(dict_df_stats.keys())
    # compile df_fdr_prop, df_assoc_fdr, df_hetero_fdr from dict_df_stats
    df_fdr_prop = pd.concat(
        [
            dict_df_stats[trait]["n_fdr_0.1"] / dict_df_stats[trait]["n_cell"]
            for trait in trait_list
        ],
        axis=1,
    ).T
    df_assoc_fdr = pd.concat(
        [dict_df_stats[trait]["assoc_mcp"] for trait in trait_list], axis=1
    ).T
    df_assoc_fdr = pd.DataFrame(
        multipletests(df_assoc_fdr.values.flatten(), method="fdr_bh")[1].reshape(
            df_assoc_fdr.shape
        ),
        index=df_assoc_fdr.index,
        columns=df_assoc_fdr.columns,
    )
    df_hetero_fdr = pd.concat(
        [dict_df_stats[trait]["assoc_mcz"] for trait in trait_list], axis=1
    ).T
    df_hetero_fdr = pd.DataFrame(
        multipletests(df_hetero_fdr.values.flatten(), method="fdr_bh")[1].reshape(
            df_hetero_fdr.shape
        ),
            index=df_hetero_fdr.index,
            columns=df_hetero_fdr.columns,
        )
    df_fdr_prop.index = trait_list
    df_assoc_fdr.index = trait_list
    df_hetero_fdr.index = trait_list
    

    df_hetero_fdr = df_hetero_fdr.applymap(lambda x: "" if x < 0.05 else "")
    df_hetero_fdr[df_assoc_fdr > 0.05] = ""

    fig, ax = plot_heatmap(
        df_fdr_prop,
        squaresize=40,
        heatmap_annot=df_hetero_fdr,
        heatmap_annot_kws={"color": "blue", "size": 8},
        heatmap_cbar_kws=dict(
            use_gridspec=False, location="top", fraction=0.03, pad=0.1, drawedges=True
        ),
        heatmap_vmin=0,
        heatmap_vmax=0.2,
        colormap_n_bin=3,
    )

    small_squares(
        ax,
        pos=[(y, x) for x, y in zip(*np.where(df_assoc_fdr < 0.05))],
        size=0.6,
        linewidth=0.5,
    )

    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=8)

    cb.ax.set_title("Prop. of sig. cells", fontsize=8)
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(1)

    plt.tight_layout()