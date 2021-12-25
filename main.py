from src.ensemble import voting_ensemble

en = voting_ensemble(load_models=False)

print(en.ensembled_prediction)
