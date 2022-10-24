
#  微調說明

最後編輯時間：2022/07/19

針對不同需求進行以下調整：

1. 定義 `vocabulary.tokenize` 函數，輸入 `string` 類型，輸出 `list` 類型。

2. 定義 `loader.process` 與 `loader.collect` 函數，兩者皆基於 table 的每個 row 逐一執行，前者是針對每個 row 的 item 進行資料處理；後者針對每個 batch 的 item 進行彙整輸出，來當作模型的 batch 輸入結構。
