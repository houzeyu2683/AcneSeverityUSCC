
#  微調說明

最後編輯時間：2022/07/19

針對不同需求進行以下調整：

1. 定義 `network.model` 物件，輸入 `batch` 類型，輸出 `batch` 類型，其中損失函數直接定義在其中。

2. 定義 `network.machine` 物件，優化器 `optimization` 跟 `schedule` 定義在此；內部方法 `learn` 、 `evaluate` 、 `infer` 很像但目的不太一樣。
   1. 方法 `learn` 目的在 backward 更新梯度。
   2. 方法 `evaluate` 目的在計算 `train` 、 `validation` 的 `loss` 或 `metric` ，用於監控當前模型的表現是否優於先前的紀錄點。
   3. 方法 `infer` 目的在針對實際的 `test` 進行推論，用於輸出預測概率以及預測結果，整合成表格輸出，目的是用於儲存測試集的預測結果或是後續 ensemble 的執行。

