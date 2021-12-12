from src.ensemble import voting_ensemble

en = voting_ensemble()

print(en.ensembled_prediction)
print(en.ensembled_eval)
