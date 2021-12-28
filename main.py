from src.ensemble import voting_ensemble

en = voting_ensemble(load_models=True)

print(en.img_avg_arr)
