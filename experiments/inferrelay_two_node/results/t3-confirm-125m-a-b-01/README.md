# 125M小prefill交错复测

固定A→B、b1/q128/gen4、priority、连续50%切分与四等分接力；case顺序(C,R,R,C)重复3轮，每case1次warmup+3次正式请求。每臂18个样本，原始值见A/raw.json和confirmation.json。

C1/R1整体中位数31.57/31.09ms；R1存在明显长尾。6个相邻C/R配对case中位数之比，描述性bootstrap95%约0.723–1.071，包含1。只有6对且可能相关，不作强统计结论；不能据约1.5%的整体中位数差声称稳健优势。
