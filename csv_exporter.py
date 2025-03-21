import os
import csv

def construct_csv_filename(experiment_params):
    """
    実験パラメータから CSV のファイル名を構築する。
    形式は
      {experimenter}_{date}_{impurity_type},{impurity_concentration}_{seed_size}_{crystallization_time}_{suspension_density}_{image_scaler}.csv
    例: Arai_241224_Mg,K=4,3_Seed300-500_30min_Mt30%_150-212μm.csv
    """
    # 各値は空白が含まれている可能性があるため、空白はアンダースコアに置換
    exp = experiment_params.get("experimenter", "").replace(" ", "_")
    date = experiment_params.get("date", "").replace(" ", "_")
    impurity_type = experiment_params.get("impurity_type", "").replace(" ", "_")
    impurity_conc = experiment_params.get("impurity_concentration", "").replace(" ", "_")
    seed_size = experiment_params.get("seed_size", "").replace(" ", "_")
    crystallization_time = experiment_params.get("crystallization_time", "").replace(" ", "_")
    suspension_density = experiment_params.get("suspension_density", "").replace(" ", "_")
    image_scaler = experiment_params.get("image_scaler", "").replace(" ", "_")
    
    filename = f"{exp}_{date}_{impurity_type},{impurity_conc}_{seed_size}_{crystallization_time}_{suspension_density}_{image_scaler}.csv"
    return filename

def export_csv(results, experiment_params, output_folder):
    """
    解析結果を CSV として output_folder 内に出力する。
    
    Parameters:
      results: dict のリスト。各 dict は少なくとも以下のキーを持つことが想定される。
               "mask_index", "area", "obb_width", "obb_height", "obb_angle"
      experiment_params: dict
         以下のキーを想定：
         "experimenter", "date", "impurity_type", "impurity_concentration",
         "seed_size", "crystallization_time", "suspension_density", "image_scaler"
      output_folder: CSV ファイルを保存するフォルダ（存在しない場合は作成）
      
    Returns:
      CSV ファイルのフルパスを返す。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = construct_csv_filename(experiment_params)
    csv_path = os.path.join(output_folder, filename)
    
    fieldnames =  ["image_filename", "particle_id", "secondary_components", "particle_type", "Lmajor [um]", "Lminor [um]"]
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return csv_path
