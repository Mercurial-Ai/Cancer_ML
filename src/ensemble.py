import os
from keras.backend import argmax, expand_dims
from keras.models import load_model
import numpy as np
from src.class_loss import class_loss
from src.confusion_matrix import confusion_matrix
from src.cancer_ml import cancer_ml
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from src.metrics import recall_m, precision_m, f1_m, BalancedSparseCategoricalAccuracy

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

        duke_independent = ['Patient ID', 'Days to MRI (From the Date of Diagnosis)', 'Manufacturer', 'Manufacturer Model Name', 'Scan Options', 'Field Strength (Tesla)', 'Patient Position During MRI', 'Image Position of Patient', 'Contrast Agent', 'Contrast Bolus Volume (mL)', 'TR (Repetition Time)', 'TE (Echo Time)',
                            'Acquisition Matrix', 'Slice Thickness', 'Rows', 'Columns', 'Flip Angle', 'FOV Computed (Field of View) in cm', 'Date of Birth (Days)', 'Menopause (at diagnosis)', 'Race and Ethnicity', 'Metastatic at Presentation (Outside of Lymph Nodes', 'ER', 'PR', 'Mol Subtype', 'Oncotype score',
                            'Staging(Tumor Size)# [T]', 'Staging(Nodes)#(Nx replaced by -1)[N]', 'Staging(Metastasis)#(Mx -replaced by -1)[M]', 'Tumor Grade', 'Position', 'Bilateral Information', 'For Other Side If Bilateral', 'Multicentric/Multifocal', 'Contralateral Breast Involvement',
                            'Lymphadenopathy or Suspicious Nodes', 'Skin/Nipple Invovlement', 'Pec/Chest Involvement', 'Age at mammo (days)', 'Breast Density', 'Shape', 'Margin', 'Architectural distortion', 'Mass Density', 'Calcifications', 'Tumor Size (cm)', 'Shape.1', 'Margin.1', 'Tumor Size (cm).1', 'Echogenicity', 'Solid'
                            'Posterior acoustic shadowing', 'Known Ovarian Status', 'Number of Ovaries In Situ', ]

        duke_dependent = ['Surgery', 'Days to Surgery (from the date of diagnosis)', 'Definitive Surgery Type', 'Clinical Response, Evaluated Through Imaging ', 'Pathologic Response to Neoadjuvant Therapy', 'Days to local recurrence (from the date of diagnosis) ', 'Days to distant recurrence(from the date of diagnosis) ', 'Days to death (from the date of diagnosis) ',
                            'Days to last local recurrence free assessment (from the date of diagnosis) ', 'Days to last distant recurrence free assemssment(from the date of diagnosis) ', 'Neoadjuvant Chemotherapy', 'Adjuvant Chemotherapy', 'Neoadjuvant Endocrine Therapy Medications ',
                            'Adjuvant Endocrine Therapy Medications ', 'Therapeutic or Prophylactic Oophorectomy as part of Endocrine Therapy ', 'Neoadjuvant Anti-Her2 Neu Therapy', 'Adjuvant Anti-Her2 Neu Therapy ', 'Received Neoadjuvant Therapy or Not', 'Pathologic response to Neoadjuvant therapy: Pathologic stage (T) following neoadjuvant therapy ',
                            'Pathologic response to Neoadjuvant therapy:  Pathologic stage (N) following neoadjuvant therapy', 'Pathologic response to Neoadjuvant therapy:  Pathologic stage (M) following neoadjuvant therapy ', 'Overall Near-complete Response:  Stricter Definition', 'Overall Near-complete Response:  Looser Definition', 'Near-complete Response (Graded Measure)']

        image_only = cancer_ml("duke", "Adjuvant Endocrine Therapy Medications ", model="cnn")

        if not load_models:

            image_only.run_model()
            image_only.test_model()

        self.image_only_models = self.load_models('data/saved_models/image_only')

        all_models = [self.image_only_models]

        image_only_test = [image_only.data_pipe.image_only.X_test, image_only.data_pipe.image_only.y_test]

        all_data = [image_only_test]

        i = 0
        all_predictions = np.array([[]])
        for models in all_models:

            # filter out empty model lists
            if len(models) >= 1:

                predictions = []

                data = all_data[i]

                testX = data[0]
                testY = data[1]

                # differentiate between image clinical and other submodels
                if type(testX) == list:
                    image_clinical_active = True
                    testX_clinical = testX[0][0]
                    testX_image = testX[0][1]

                    first_dim = len(testX_clinical)
                    second_dim = len(testX_clinical[0])

                    print("first dim clinical:", first_dim)
                    print("second dim clinical:", second_dim)

                    array_testX_clinical = np.empty(shape=(first_dim, second_dim), dtype=np.float16)

                    j = 0
                    for example in testX_clinical:
                        example = np.asarray(example, dtype=np.float16)
                        array_testX_clinical[j] = example
                        j = j + 1

                    first_dim = len(testX_image)
                    second_dim = len(testX_image[0])

                    print("first dim image:", first_dim)
                    print("second dim image:", second_dim)

                    array_testX_image = np.empty(shape=(first_dim, second_dim, second_dim, 1), dtype=np.float16)

                    j = 0
                    for example in testX_image:
                        example = np.asarray(example, dtype=np.float16)
                        array_testX_image[j] = example
                        j = j + 1

                    testX = [array_testX_clinical, array_testX_image]

                    # make sure model input shape matches x shape
                    print(models[0].layers[0].get_output_at(0).get_shape())
                    if models[0].layers[0].get_output_at(0).get_shape()[1:] == testX[0].shape[1:] or models[0].layers[0].get_output_at(0).get_shape()[1:] == testX[1].shape[1:]:

                        # differentiate between image-clinical and other submodels
                        if type(testX) != list:
                            for example in testX:
                                ensemble_prediction = self.predict(example, models)
                                predictions.append(ensemble_prediction)

                            predictions = np.flip(predictions)
                        else:
                            for j in range(testX[0].shape[0]):
                                clinical_example = testX[0][j]
                                img_example = testX[1][j]

                                example = [clinical_example, img_example]
                                ensemble_prediction = self.predict(example, models)
                                predictions.append(ensemble_prediction)

                                ensemble_prediction = np.flip(ensemble_prediction)
                else:
                    image_clinical_active = False
                    # make sure model input shape matches x shape
                    if models[0].layers[0].get_output_at(0).get_shape() == testX.shape[1:]:

                        # differentiate between image-clinical and other submodels
                        if type(testX) != list:
                            for example in testX:
                                ensemble_prediction = self.predict(example, models)
                                predictions.append(ensemble_prediction)

                        else:
                            for j in range(testX[0].shape[0]):
                                clinical_example = testX[0][j]
                                img_example = testX[1][j]

                                clinical_example = np.expand_dims(clinical_example, 0)
                                img_example = np.expand_dims(img_example, 0)

                                example = [clinical_example, img_example]
                                ensemble_prediction = self.predict(example, models)
                                predictions.append(ensemble_prediction)

                all_predictions = np.append(all_predictions, predictions)

            if len(all_predictions) != 0:

                all_predictions = np.expand_dims(all_predictions, 0)

                if all_predictions.shape[0] != testY.shape[0]:
                    all_predictions = np.reshape(all_predictions, (testY.shape[0], -1))

                confusion_matrix(testY, all_predictions)

            i = i + 1

        self.ensembled_prediction = all_predictions

        duke_image_predictions = []
        # if multi target
        if len(self.ensembled_prediction[0].shape) != 1:
            for prediction in self.ensembled_prediction:
                if prediction.shape[-1] == 24:
                    duke_image_predictions.append(prediction)

            duke_image_predictions = np.concatenate(duke_image_predictions, axis=0)

        else:
            duke_image_predictions = self.ensembled_prediction

        duke_image_true = all_data[3][1]

        accuracy, f1, recall, balanced_acc = self.eval(duke_image_predictions, duke_image_true)

        if len(self.ensembled_prediction[0].shape) != 1:
            accuracy = dict(zip(duke_dependent, accuracy))
            f1 = dict(zip(duke_dependent, f1))
            recall = dict(zip(duke_dependent, recall))
            balanced_acc = dict(zip(duke_dependent, balanced_acc))

        print("Accuracy:", accuracy)
        print("F1:", float(f1))
        print("Recall:", float(recall))
        print("Balanced Accuracy:", float(balanced_acc))

    def load_models(self, model_dir):

        model_names = os.listdir(model_dir)

        model_paths = list()
        for name in model_names:
            full_path = os.path.join(model_dir, name)
            model_paths.append(full_path)

        models = list()
        for path in model_paths:
            model = load_model(path, custom_objects={"loss":class_loss, "f1_m": f1_m, "recall_m": recall_m, "precision_m": precision_m, 'BalancedSparseCategoricalAccuracy': BalancedSparseCategoricalAccuracy})
            models.append(model)

        return models

    def predict(self, testX, models):

        if type(testX) == list:
            i = 0
            for array in testX:
                array = np.expand_dims(array, 0)
                testX[i] = array

                i = i + 1
        else:
            testX = np.expand_dims(testX, 0)

        y = []
        for model in models:
            print(model.summary())
            prediction = model.predict(testX)
            y.append(prediction)

        results = np.concatenate(y, axis=0)

        if len(models) > 1:
            rounded_results = results.round()
            ensembled_results = np.empty(shape=(rounded_results.shape[0]), dtype=np.float16)
            for i in range(rounded_results.shape[0]):
                var = rounded_results[i]
                result = np.bincount(var).argmax()
                ensembled_results[i] = result

            print("ensembled result:", ensembled_results)

            results = ensembled_results

        return results

    def eval(self, prediction, testY):

        if type(testY) == pd.DataFrame:
            testY = testY.to_numpy()

        if prediction.shape[0] != testY.shape[0]:
            prediction = np.reshape(prediction, (testY.shape[0], -1))

        testY = testY.astype(np.float32)
        prediction = prediction.round().astype(np.float32)

        if len(testY.shape) == 1:
            accuracies = accuracy_score(testY, prediction)
            f1_scores = f1_m(testY, prediction)
            recall_scores = recall_m(testY, prediction)
            balanced_acc_scores = balanced_accuracy_score(testY, prediction)

        else:
            accuracies = []
            f1_scores = []
            recall_scores = []
            balanced_acc_scores = []
            for j in range(testY.shape[-1]):

                pred = prediction[:, j].round().astype(np.float32)
                true = testY[:, j].round().astype(np.float32)

                # try installing earlier version of tf
                accuracy = accuracy_score(true, pred)
                f1_score = f1_m(true, pred)
                recall = recall_m(true, pred)
                balanced_acc = balanced_accuracy_score(true, pred)

                accuracies.append(accuracy)
                f1_scores.append(f1_score)
                recall_scores.append(recall)
                balanced_acc_scores.append(balanced_acc)

        return accuracies, f1_scores, recall_scores, balanced_acc_scores
