from src.ensemble import voting_ensemble

en = voting_ensemble(load_models=False)

print(en.img_avg_arr)
