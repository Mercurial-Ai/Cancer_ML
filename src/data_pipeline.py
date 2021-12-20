from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math

from src.image_tools.import_numpy import import_numpy
from src.image_tools.remove_ids import remove_ids
from src.tokenize_dataset import tokenize_dataset

class data_pod:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.y_train = None
        self.y_test = None
        self.y_val = None

class data_pipeline:

    def __init__(self, clinical_path, image_path, target):
        self.clinical_path = clinical_path
        self.image_path = image_path
        self.target = target

        self.only_clinical = data_pod()
        self.image_clinical = data_pod()
        self.image_only = data_pod()

    def load_data(self):
        self.df = pd.read_csv(self.clinical_path, low_memory=False)

        metabric_independent = ['age_at_diagnosis', 'cancer_type', 'cancer_type_detailed', 'cellularity', 'pam50_+_claudin-low_subtype', 'cohort', 'er_status_measured_by_ihc', 'er_status', 'neoplasm_histologic_grade', 'her2_status_measured_by_snp6', 'her2_status', 'tumor_other_histologic_subtype',
                                'inferred_menopausal_state', 'integrative_cluster', 'primary_tumor_laterality', 'lymph_nodes_examined_positive', 'mutation_count', 'nottingham_prognostic_index', 'oncotree_code', 'pr_status', '3-gene_classifier_subtype', 'tumor_size', 'tumor_stage', 'brca1', 'brca2', 
                                'palb2', 'pten', 'tp53', 'atm', 'cdh1', 'chek2', 'nbn', 'nf1', 'stk11', 'bard1', 'mlh1', 'msh2', 'msh6', 'pms2', 'epcam', 'rad51c', 'rad51d', 'rad50', 'rb1', 'rbl1', 'rbl2', 'ccna1', 'ccnb1', 'cdk1', 'ccne1', 'cdk2', 'cdc25a', 'ccnd1', 'cdk4', 'cdk6', 'ccnd2', 'cdkn2a', 
                                'cdkn2b', 'myc', 'cdkn1a', 'cdkn1b', 'e2f1', 'e2f2', 'e2f3', 'e2f4', 'e2f5', 'e2f6', 'e2f7', 'e2f8', 'src', 'jak1', 'jak2', 'stat1', 'stat2', 'stat3', 'stat5a', 'stat5b', 'mdm2', 'tp53bp1', 'adam10', 'adam17', 'aph1a', 'aph1b', 'arrdc1', 'cir1', 'ctbp1', 'ctbp2', 'cul1', 
                                'dll1', 'dll3', 'dll4', 'dtx1', 'dtx2', 'dtx3', 'dtx4', 'ep300', 'fbxw7', 'hdac1', 'hdac2', 'hes1', 'hes5', 'heyl', 'itch', 'jag1', 'jag2', 'kdm5a', 'lfng', 'maml1', 'maml2', 'maml3', 'ncor2', 'ncstn', 'notch1', 'notch2', 'notch3', 'nrarp', 'numb', 'numbl', 'psen1', 
                                'psen2', 'psenen', 'rbpj', 'rbpjl', 'rfng', 'snw1', 'spen', 'hes2', 'hes4', 'hes7', 'hey1', 'hey2', 'acvr1', 'acvr1b', 'acvr1c', 'acvr2a', 'acvr2b', 'acvrl1', 'akt1', 'akt1s1', 'akt2', 'apaf1', 'arl11', 'atr', 'aurka', 'bad', 'bcl2', 'bcl2l1', 'bmp10', 'bmp15', 'bmp2', 
                                'bmp3', 'bmp4', 'bmp5', 'bmp6', 'bmp7', 'bmpr1a', 'bmpr1b', 'bmpr2', 'braf', 'casp10', 'casp3', 'casp6', 'casp7', 'casp8', 'casp9', 'chek1', 'csf1', 'csf1r', 'cxcl8', 'cxcr1', 'cxcr2', 'dab2', 'diras3', 'dlec1', 'dph1', 'egfr', 'eif4e', 'eif4ebp1', 'eif5a2', 'erbb2', 'erbb3', 
                                'erbb4', 'fas', 'fgf1', 'fgfr1', 'folr1', 'folr2', 'folr3', 'foxo1', 'foxo3', 'gdf11', 'gdf2', 'gsk3b', 'hif1a', 'hla-g', 'hras', 'igf1', 'igf1r', 'inha', 'inhba', 'inhbc', 'itgav', 'itgb3', 'izumo1r', 'kdr', 'kit', 'kras', 'map2k1', 'map2k2', 'map2k3', 'map2k4', 'map2k5', 
                                'map3k1', 'map3k3', 'map3k4', 'map3k5', 'mapk1', 'mapk12', 'mapk14', 'mapk3', 'mapk4', 'mapk6', 'mapk7', 'mapk8', 'mapk9', 'mdc1', 'mlst8', 'mmp1', 'mmp10', 'mmp11', 'mmp12', 'mmp13', 'mmp14', 'mmp15', 'mmp16', 'mmp17', 'mmp19', 'mmp2', 'mmp21', 'mmp23b', 'mmp24', 'mmp25', 
                                'mmp26', 'mmp27', 'mmp28', 'mmp3', 'mmp7', 'mmp9', 'mtor', 'nfkb1', 'nfkb2', 'opcml', 'pdgfa', 'pdgfb', 'pdgfra', 'pdgfrb', 'pdpk1', 'peg3', 'pik3ca', 'pik3r1', 'pik3r2', 'plagl1', 'ptk2', 'rab25', 'rad51', 'raf1', 'rassf1', 'rheb', 'rictor', 'rps6', 'rps6ka1', 'rps6ka2', 
                                'rps6kb1', 'rps6kb2', 'rptor', 'slc19a1', 'smad1', 'smad2', 'smad3', 'smad4', 'smad5', 'smad6', 'smad7', 'smad9', 'sptbn1', 'terc', 'tert', 'tgfb1', 'tgfb2', 'tgfb3', 'tgfbr1', 'tgfbr2', 'tgfbr3', 'tsc1', 'tsc2', 'vegfa', 'vegfb', 'wfdc2', 'wwox', 'zfyve9', 'arid1a', 'arid1b', 
                                'cbfb', 'gata3', 'kmt2c', 'kmt2d', 'myh9', 'ncor1', 'pde4dip', 'ptprd', 'ros1', 'runx1', 'tbx3', 'abcb1', 'abcb11', 'abcc1', 'abcc10', 'bbc3', 'bmf', 'cyp2c8', 'cyp3a4', 'fgf2', 'fn1', 'map2', 'map4', 'mapt', 'nr1i2', 'slco1b3', 'tubb1', 'tubb4a', 'tubb4b', 'twist1', 'adgra2', 
                                'afdn', 'aff2', 'agmo', 'agtr2', 'ahnak', 'ahnak2', 'akap9', 'alk', 'apc', 'arid2', 'arid5b', 'asxl1', 'asxl2', 'bap1', 'bcas3', 'birc6', 'cacna2d3', 'ccnd3', 'chd1', 'clk3', 'clrn2', 'col12a1', 'col22a1', 'col6a3', 'ctcf', 'ctnna1', 'ctnna3', 'dnah11', 'dnah2', 'dnah5', 'dtwd2', 
                                'fam20c', 'fanca', 'fancd2', 'flt3', 'foxp1', 'frmd3', 'gh1', 'gldc', 'gpr32', 'gps2', 'hdac9', 'herc2', 'hist1h2bc', 'kdm3a', 'kdm6a', 'klrg1', 'l1cam', 'lama2', 'lamb3', 'large1', 'ldlrap1', 'lifr', 'lipi', 'magea8', 'map3k10', 'map3k13', 'men1', 'mtap', 'muc16', 'myo1a', 'myo3a', 
                                'ncoa3', 'nek1', 'nf2', 'npnt', 'nr2f1', 'nr3c1', 'nras', 'nrg3', 'nt5e', 'or6a2', 'palld', 'pbrm1', 'ppp2cb', 'ppp2r2a', 'prkacg', 'prkce', 'prkcq', 'prkcz', 'prkg1', 'prps2', 'prr16', 'ptpn22', 'ptprm', 'rasgef1b', 'rpgr', 'ryr2', 'sbno1', 'setd1a', 'setd2', 'setdb1', 'sf3b1', 
                                'sgcd', 'shank2', 'siah1', 'sik1', 'sik2', 'smarcb1', 'smarcc1', 'smarcc2', 'smarcd1', 'spaca1', 'stab2', 'stmn2', 'syne1', 'taf1', 'taf4b', 'tbl1xr1', 'tg', 'thada', 'thsd7a', 'ttyh1', 'ubr5', 'ush2a', 'usp9x', 'utrn', 'zfp36l1', 'ackr3', 'akr1c1', 'akr1c2', 'akr1c3', 'akr1c4', 
                                'akt3', 'ar', 'bche', 'cdk8', 'cdkn2c', 'cyb5a', 'cyp11a1', 'cyp11b2', 'cyp17a1', 'cyp19a1', 'cyp21a2', 'cyp3a43', 'cyp3a5', 'cyp3a7', 'ddc', 'hes6', 'hsd17b1', 'hsd17b10', 'hsd17b11', 'hsd17b12', 'hsd17b13', 'hsd17b14', 'hsd17b2', 'hsd17b3', 'hsd17b4', 'hsd17b6', 'hsd17b7', 
                                'hsd17b8', 'hsd3b1', 'hsd3b2', 'hsd3b7', 'mecom', 'met', 'ncoa2', 'nrip1', 'pik3r3', 'prkci', 'prkd1', 'ran', 'rdh5', 'sdc4', 'serpini1', 'shbg', 'slc29a1', 'sox9', 'spry2', 'srd5a1', 'srd5a2', 'srd5a3', 'st7', 'star', 'tnk2', 'tulp4', 'ugt2b15', 'ugt2b17', 'ugt2b7', 
                                'pik3ca_mut', 'tp53_mut', 'muc16_mut', 'ahnak2_mut', 'kmt2c_mut', 'syne1_mut', 'gata3_mut', 'map3k1_mut', 'ahnak_mut', 'dnah11_mut', 'cdh1_mut', 'dnah2_mut', 'kmt2d_mut', 'ush2a_mut', 'ryr2_mut', 'dnah5_mut', 'herc2_mut', 'pde4dip_mut', 'akap9_mut', 'tg_mut', 'birc6_mut', 
                                'utrn_mut', 'tbx3_mut', 'col6a3_mut', 'arid1a_mut', 'lama2_mut', 'notch1_mut', 'cbfb_mut', 'ncor2_mut', 'col12a1_mut', 'col22a1_mut', 'pten_mut', 'akt1_mut', 'atr_mut', 'thada_mut', 'ncor1_mut', 'stab2_mut', 'myh9_mut', 'runx1_mut', 'nf1_mut', 'map2k4_mut', 'ros1_mut', 
                                'lamb3_mut', 'arid1b_mut', 'erbb2_mut', 'sf3b1_mut', 'shank2_mut', 'ep300_mut', 'ptprd_mut', 'usp9x_mut', 'setd2_mut', 'setd1a_mut', 'thsd7a_mut', 'afdn_mut', 'erbb3_mut', 'rb1_mut', 'myo1a_mut', 'alk_mut', 'fanca_mut', 'adgra2_mut', 'ubr5_mut', 'pik3r1_mut', 'myo3a_mut', 
                                'asxl2_mut', 'apc_mut', 'ctcf_mut', 'asxl1_mut', 'fancd2_mut', 'taf1_mut', 'kdm6a_mut', 'ctnna3_mut', 'brca1_mut', 'ptprm_mut', 'foxo3_mut', 'usp28_mut', 'gldc_mut', 'brca2_mut', 'cacna2d3_mut', 'arid2_mut', 'aff2_mut', 'lifr_mut', 'sbno1_mut', 'kdm3a_mut', 'ncoa3_mut', 
                                'bap1_mut', 'l1cam_mut', 'pbrm1_mut', 'chd1_mut', 'jak1_mut', 'setdb1_mut', 'fam20c_mut', 'arid5b_mut', 'egfr_mut', 'map3k10_mut', 'smarcc2_mut', 'erbb4_mut', 'npnt_mut', 'nek1_mut', 'agmo_mut', 'zfp36l1_mut', 'smad4_mut', 'sik1_mut', 'casp8_mut', 'prkcq_mut', 'smarcc1_mut', 
                                'palld_mut', 'dcaf4l2_mut', 'bcas3_mut', 'cdkn1b_mut', 'gps2_mut', 'men1_mut', 'stk11_mut', 'sik2_mut', 'ptpn22_mut', 'brip1_mut', 'flt3_mut', 'nrg3_mut', 'fbxw7_mut', 'ttyh1_mut', 'taf4b_mut', 'or6a2_mut', 'map3k13_mut', 'hdac9_mut', 'prkacg_mut', 'rpgr_mut', 'large1_mut', 
                                'foxp1_mut', 'clk3_mut', 'prkcz_mut', 'lipi_mut', 'ppp2r2a_mut', 'prkce_mut', 'gh1_mut', 'gpr32_mut', 'kras_mut', 'nf2_mut', 'chek2_mut', 'ldlrap1_mut', 'clrn2_mut', 'acvrl1_mut', 'agtr2_mut', 'cdkn2a_mut', 'ctnna1_mut', 'magea8_mut', 'prr16_mut', 'dtwd2_mut', 'akt2_mut', 
                                'braf_mut', 'foxo1_mut', 'nt5e_mut', 'ccnd3_mut', 'nr3c1_mut', 'prkg1_mut', 'tbl1xr1_mut', 'frmd3_mut', 'smad2_mut', 'sgcd_mut', 'spaca1_mut', 'rasgef1b_mut', 'hist1h2bc_mut', 'nr2f1_mut', 'klrg1_mut', 'mbl2_mut', 'mtap_mut', 'ppp2cb_mut', 'smarcd1_mut', 'nras_mut', 'ndfip1_mut', 
                                'hras_mut', 'prps2_mut', 'smarcb1_mut', 'stmn2_mut', 'siah1_mut']

        metabric_dependent = ['type_of_breast_surgery', 'chemotherapy', 'hormone_therapy', 'overall_survival_months', 'overall_survival', 'radio_therapy', 'death_from_cancer']

        self.clinical_ids = self.df[list(self.df.columns)[0]]

        self.df = self.df.set_index(list(self.df.columns)[0])

        self.df = tokenize_dataset(self.df)

        # if image path = None, dataset should be clinical only and imagery does not need to be imported
        if self.image_path != None:
            self.img_array = import_numpy(self.image_path, self.clinical_ids)

            self.slice_data()

            self.image_ids = self.img_array[:, -1]

            # get patients in clinical data with ids that correspond with image ids
            self.filtered_df = self.df.loc[self.image_ids]

            self.concatenated_array = self.concatenate_image_clinical()

            self.partition_image_clinical_data()
            self.partition_image_only_data()

        self.partition_clinical_only_data()

    def concatenate_image_clinical(self):

        clinical_array = self.filtered_df.to_numpy()
        concatenated_array = np.concatenate((clinical_array, self.img_array), axis=1)

        return concatenated_array

    def split_data(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=84)

        # split test data into validation and test
        X_test, X_val = train_test_split(X_test, test_size=0.5, random_state=73)
        y_test, y_val = train_test_split(y_test, test_size=0.5, random_state=35)

        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def partition_clinical_only_data(self):
        
        x = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        # normalize data
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        self.only_clinical.X_train = X_train
        self.only_clinical.X_test = X_test
        self.only_clinical.y_train = y_train
        self.only_clinical.y_test = y_test
        self.only_clinical.X_val = X_val
        self.only_clinical.y_val = y_val

    def split_modalities(self, x):
        clinical_x = x[:, :75]
        image_x = x[:, 75:]

        # unflatten images in image_x
        unflattened_array = np.empty(shape=(image_x.shape[0], int(math.sqrt(image_x.shape[-1])), int(math.sqrt(image_x.shape[-1])), 1), dtype=np.int8)
        i = 0
        for image in image_x:
            image = np.reshape(image, (1, 512, 512, 1))
            unflattened_array[i] = image

            i = i + 1

        return clinical_x, unflattened_array

    def partition_image_clinical_data(self):

        target_index = self.df.columns.get_loc(self.target)

        self.concatenated_array = self.concatenated_array.astype(np.int8)

        x = np.delete(self.concatenated_array, target_index, axis=1)
        y = self.concatenated_array[:, target_index]

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        # normalize data
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        X_train = [self.split_modalities(X_train)]
        X_test = [self.split_modalities(X_test)]
        X_val = [self.split_modalities(X_val)]

        self.image_clinical.X_train = X_train
        self.image_clinical.X_test = X_test
        self.image_clinical.y_train = y_train
        self.image_clinical.y_test = y_test
        self.image_clinical.X_val = X_val
        self.image_clinical.y_val = y_val

    def partition_image_only_data(self):

        x = self.img_array
        y = self.filtered_df[self.target]

        x = remove_ids(x)

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        # reshape back into 2d images
        X_train = np.reshape(X_train, (X_train.shape[0], int(math.sqrt(X_train.shape[1])), int(math.sqrt(X_train.shape[1]))))
        X_test = np.reshape(X_test, (X_test.shape[0], int(math.sqrt(X_test.shape[1])), int(math.sqrt(X_test.shape[1]))))
        X_val = np.reshape(X_val, (X_val.shape[0], int(math.sqrt(X_val.shape[1])), int(math.sqrt(X_val.shape[1]))))

        # add additional dimension at the end of the shape to each partition
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

        self.image_only.X_train = X_train
        self.image_only.X_test = X_test
        self.image_only.X_val = X_val
        self.image_only.y_train = y_train
        self.image_only.y_test = y_test
        self.image_only.y_val = y_val

    def slice_data(self):
        slice_size = 0.31

        self.img_array = self.img_array[0:int(round(self.img_array.shape[0]*slice_size, 0))]
