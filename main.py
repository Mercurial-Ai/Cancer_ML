from src.ensemble import voting_ensemble

en = voting_ensemble(load_models=True)

print(en.ensembled_prediction)
print(en.ensembled_eval)
