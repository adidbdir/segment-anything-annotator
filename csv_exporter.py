import os
import csv
from datetime import datetime
from pathlib import Path

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

def get_max_particle_id_from_csv(filepath):
    """
    CSVファイルから最大の一次粒子IDを取得する
    
    Args:
        filepath (str or Path): CSVファイルのパス
    
    Returns:
        int: 最大の一次粒子ID（ファイルが存在しない場合や読み取れない場合は0）
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return 0
    
    max_id = 0
    try:
        with filepath.open('r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 一次粒子IDカラムの値を取得
                particle_id_str = row.get('一次粒子ID', '')
                
                # 二次粒子の場合はカンマ区切りの値を含むため、スキップ
                if ',' in str(particle_id_str):
                    continue
                    
                try:
                    particle_id = int(particle_id_str)
                    max_id = max(max_id, particle_id)
                except (ValueError, TypeError):
                    # 数値変換できない場合は無視
                    pass
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中に問題が発生しました - {e}")
    
    return max_id

def export_csv(results, experiment_params, output_dir):
    """
    パーティクル解析結果をCSVファイルにエクスポートする関数
    追記モードをサポート：既存ファイルがあれば追記、なければ新規作成
    
    Args:
        results (list): 解析結果のリスト。各要素は辞書形式でフィールド名と値を持つ
        experiment_params (dict): 実験パラメータ情報の辞書
        output_dir (str): 出力ディレクトリのパス
    """
    
    filename = construct_csv_filename(experiment_params)
    output_path = Path(output_dir)
    filepath = output_path / filename
    
    # ヘッダーフィールドの定義 - ふるい_500_,____.csv に合わせる
    fieldnames = [
        '画像ファイル名', '一次粒子ID', '二次粒子ID', '粒子形態', 
        'Lmajor[um]', 'Lminor[um]', 'L[um]', 'Lmean[um]', 
        'n', 'Agg.', 'Area[um^2]'
    ]
    
    # 結果データの変換とフィールド名マッピング
    mapped_results = []
    for result in results:
        mapped_result = {
            '画像ファイル名': result.get('image_filename', ''),
            '一次粒子ID': result.get('particle_id', ''),
            '二次粒子ID': result.get('secondary_id', ''),
            '粒子形態': result.get('particle_type', ''),
            'Lmajor[um]': result.get('Lmajor [um]', ''),
            'Lminor[um]': result.get('Lminor [um]', ''),
            'L[um]': result.get('L[um]', ''),
            'Lmean[um]': result.get('Lmean[um]', ''),
            'n': result.get('n', ''),
            'Agg.': result.get('Agg.', ''),
            'Area[um^2]': result.get('Area[um^2]', '')
        }
        mapped_results.append(mapped_result)
    
    # ファイルが存在するかチェック
    file_exists = filepath.exists()
    
    # CSVファイルへの書き込み（追記または新規作成）
    mode = 'a' if file_exists else 'w'
    with filepath.open(mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 新規ファイルの場合のみヘッダーを書き込み
        if not file_exists:
            writer.writeheader()
        
        # データ行の書き込み
        writer.writerows(mapped_results)
    
    # print(f"CSV data {'appended to' if file_exists else 'exported to'}: {filepath}")
    return str(filepath)
