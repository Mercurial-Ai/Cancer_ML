import os
from keras.backend import argmax
from keras.models import load_model
from scipy.sparse import data
from sklearn.metrics import accuracy_score
import numpy as np
from src.cancer_ml import cancer_ml

class voting_ensemble:

    # input iterables containing [x, y] for each data form
    def __init__(self, load_models):

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

        clinical = cancer_ml("metabric", metabric_dependent, model="clinical_only")
        image_clinical = cancer_ml("duke", "Adjuvant Chemotherapy", model="image_clinical")
        image_only = cancer_ml("duke", "Adjuvant Chemotherapy", model="cnn")

        if not load_models:
            clinical.run_model()
            clinical.test_model()

            image_clinical.run_model()
            image_clinical.test_model()

            image_only.run_model()
            image_only.test_model()

        self.clinical_models = self.load_models('data/saved_models/clinical')
        self.image_clinical_models = self.load_models('data/saved_models/image_clinical')
        self.image_only_models = self.load_models('data/saved_models/image_only')

        all_models = [self.clinical_models, self.image_clinical_models, self.image_only_models]

        clinical_test = [clinical.data_pipe.only_clinical.X_test, clinical.data_pipe.only_clinical.y_test]
        image_clinical_test = [image_clinical.data_pipe.image_clinical.X_test, image_clinical.data_pipe.image_clinical.y_test]
        image_only_test = [image_only.data_pipe.image_only.X_test, image_only.data_pipe.image_only.y_test]

        all_data = [clinical_test, image_clinical_test, image_only_test]

        i = 0
        predictions = []
        evals = []
        for models in all_models:

            # filter out empty model lists
            if len(models) >= 1:
                data = all_data[i]

                testX = data[0]
                testY = data[1]

                ensemble_prediction = self.predict(testX, models)
                predictions.append(ensemble_prediction)

                ensemble_eval = self.evaluate_models(testX, testY, models)
                evals.append(ensemble_eval)

            i = i + 1

        ensembled_predictions = []
        ensembled_evals = []
            
        # find average of predictions and evals for ensembled values
        for prediction in predictions:
            ensembled_predictions.append(float(np.mean(prediction)))

        for eval in evals:
            ensembled_evals.append(float(np.mean(eval)))

        self.ensembled_prediction = sum(ensembled_predictions) / len(ensembled_predictions)
        self.ensembled_eval = sum(ensembled_evals) / len(ensembled_evals)
        
    def load_models(self, model_dir):

        model_names = os.listdir(model_dir)

        model_paths = list()
        for name in model_names:
            full_path = os.path.join(model_dir, name)
            model_paths.append(full_path)

        models = list()
        for path in model_paths:
            model = load_model(path)
            models.append(model)

        return models

    def predict(self, testX, models):

        y = [model.predict(testX) for model in models]
        y = np.array(y)

        sum = np.sum(y, axis=0)

        result = argmax(sum, axis=1)

        return result

    def evaluate_models(self, testX, testY, models):

        y = self.predict(testX, models)

        return accuracy_score(testY, y)
