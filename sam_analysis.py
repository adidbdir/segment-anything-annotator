# sam_analysis.py
import pandas as pd
import numpy as np

# 元データの保持クラス
class SAMData:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)

# SAMの結果を集計したもの
class AggregatedSAMData:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._aggregate()

    def _aggregate(self):
        self.df['L[um]'] = np.sqrt(self.df['Lmajor[um]'] * self.df['Lminor[um]'])
        self.df['Area[um^2]'] = self.df['Lmajor[um]'] * self.df['Lminor[um]']
        
        # 二次粒子側の集計
        secondary_particles = self.df[self.df['粒子形態'] == 'secondary']
        for idx, row in secondary_particles.iterrows():
            primary_ids = str(row['一次粒子ID']).split(',')
            primary_particles = self.df[self.df['一次粒子ID'].isin(primary_ids) & (self.df['粒子形態'] == 'primary')]

            Lp_mean = primary_particles['L[um]'].mean()
            n = primary_particles.shape[0]
            Agg = row['L[um]'] / Lp_mean if Lp_mean else np.nan
            Area_sum = primary_particles['Area[um^2]'].sum()

            self.df.at[idx, 'Lp,mean[um]'] = Lp_mean
            self.df.at[idx, 'n'] = n
            self.df.at[idx, 'Agg.[um]'] = Agg
            self.df.at[idx, 'Area[um^2]'] = max(row['Area[um^2]'], Area_sum)

# SAMの結果をさらに集計したもの
class SecondaryAggregatedData:
    def __init__(self, aggregated_df: pd.DataFrame, bin_width: float = 50):
        self.bin_width = bin_width
        self.aggregated_df = aggregated_df
        self.binned_df = self._secondary_aggregate()

    def _secondary_aggregate(self):
        sec_df = self.aggregated_df[self.aggregated_df['粒子形態'] == 'secondary'].copy()
        sec_df['L_bin'] = (sec_df['L[um]'] // self.bin_width) * self.bin_width

        result = sec_df.groupby('L_bin').agg({
            'Lp,mean[um]': 'mean',
            'Agg.[um]': 'mean',
            'n': 'sum'
        }).reset_index()

        return result.rename(columns={'Lp,mean[um]': 'Lp_mean_bin_avg', 'Agg.[um]': 'Agg_bin_avg'})

# 処理クラス
class SAMDataProcessor:
    def __init__(self, filepath: str, bin_width: float = 50):
        self.sam_data = SAMData(filepath)
        self.aggregated_data = AggregatedSAMData(self.sam_data.df)
        self.secondary_aggregated_data = SecondaryAggregatedData(self.aggregated_data.df, bin_width)

    def save_primary_aggregated(self, filepath: str):
        self.aggregated_data.df.to_csv(filepath, index=False)

    def save_secondary_aggregated(self, filepath: str):
        self.secondary_aggregated_data.binned_df.to_csv(filepath, index=False)

    # 修正: 全ての結果を一つのCSVにまとめて出力するメソッド
    def save_all_results(self, filepath: str):
        # 元のデータフレームを取得
        original_df = self.sam_data.df.copy()
        
        # 集計結果を取得
        aggregated_df = self.aggregated_data.df.copy()
        
        # 二次集計結果を取得
        secondary_df = self.secondary_aggregated_data.binned_df.copy()

        # 二次集計結果に元のデータを結合
        combined_df = original_df.merge(aggregated_df, on='一次粒子ID', how='left', suffixes=('', '_agg'))
        combined_df = combined_df.merge(secondary_df, left_on='L_bin', right_on='L_bin', how='left', suffixes=('', '_sec'))

        # 結合したデータフレームをCSVに保存
        combined_df.to_csv(filepath, index=False)

# 実行例
if __name__ == '__main__':
    processor = SAMDataProcessor('mock_sam_results.csv', bin_width=50)
    processor.save_primary_aggregated('aggregated_result.csv')
    processor.save_secondary_aggregated('secondary_aggregated_result.csv')
    
    # 追加: 全ての結果を一つのCSVにまとめて出力
    processor.save_all_results('all_results_combined.csv')  # 全ての結果を保存

    # # 他のプログラムでの使用例
    # from sam_analysis import SAMDataProcessor

    # # 入力CSVのパス
    # input_csv = 'path/to/your/sam_results.csv'

    # # 出力CSVのパス
    # primary_aggregated_csv = 'path/to/save/primary_aggregated.csv'
    # secondary_aggregated_csv = 'path/to/save/secondary_aggregated.csv'

    # # SAMデータの処理（階級幅50umで二次集計）
    # processor = SAMDataProcessor(input_csv, bin_width=50)

    # # SAMの結果を集計したCSVを出力
    # processor.save_primary_aggregated(primary_aggregated_csv)

    # # SAMの結果をさらに集計したCSVを出力
    # processor.save_secondary_aggregated(secondary_aggregated_csv)
